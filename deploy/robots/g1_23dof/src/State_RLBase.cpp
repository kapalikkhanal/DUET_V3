#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

#include <algorithm>
#include <cmath>
#include <mutex>

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    // Safety guards. Each one drops to Passive (damping) and logs why.
    {
        auto dcfg = YAML::LoadFile(policy_dir / "params" / "deploy.yaml");
        if (dcfg["safety"]) {
            auto s = dcfg["safety"];
            auto get = [&](const char* key, float & v) { if (s[key]) v = s[key].as<float>(); };
            get("tilt_roll_deg", tilt_roll_deg_);
            get("tilt_back_deg", tilt_back_deg_);
            get("tilt_forward_deg", tilt_fwd_deg_);
            get("tilt_lean_margin_deg", tilt_lean_margin_deg_);
            get("tilt_hold_s", tilt_hold_s_);
            get("joint_speed_rms", speed_rms_limit_);
            get("start_max_tilt_deg", start_max_tilt_deg_);
            get("soft_start_s", soft_start_s_);
        }
        spdlog::info("Safety: Passive if tilt > {:.0f} deg sideways, {:.0f} back, {:.0f} forward (lean + {:.0f}); "
                     "leg joint speed RMS > {:.0f} rad/s; start refused above {:.0f} deg; soft start {:.1f} s",
                     tilt_roll_deg_, tilt_back_deg_, tilt_fwd_deg_, tilt_lean_margin_deg_,
                     speed_rms_limit_, start_max_tilt_deg_, soft_start_s_);
    }
    FSMState::lowstate->set_timeout_ms(300);  // no robot state for 0.3 s -> Passive (all states)
    this->registered_checks.emplace_back(
        std::make_pair([&]()->bool{ return start_refused(); }, FSMStringMap.right.at("Passive")));
    this->registered_checks.emplace_back(
        std::make_pair([&]()->bool{ return tilt_guard(); }, FSMStringMap.right.at("Passive")));
    this->registered_checks.emplace_back(
        std::make_pair([&]()->bool{ return speed_guard(); }, FSMStringMap.right.at("Passive")));

    // Runaway-policy guard: drop to Passive (damping) if the raw policy
    // output stays above guard_max_abs_action_ for guard_hold_s_. Time-based
    // because the checks run at the FSM rate, not the 50 Hz policy rate.
    auto deploy_cfg = YAML::LoadFile(policy_dir / "params" / "deploy.yaml");
    if (deploy_cfg["safety"]) {
        auto s = deploy_cfg["safety"];
        if (s["max_abs_action"]) guard_max_abs_action_ = s["max_abs_action"].as<float>();
        if (s["max_abs_action_hold_s"]) guard_hold_s_ = s["max_abs_action_hold_s"].as<float>();
    }
    spdlog::info("RL guard: Passive if |raw action| > {:.1f} for {:.0f} ms",
                 guard_max_abs_action_, 1000.0f * guard_hold_s_);
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return action_runaway(); },
            FSMStringMap.right.at("Passive")
        )
    );

    // Subscribe to upper-body (arm) targets published by the IL / GR00T policy on
    // a dedicated topic (reuses the LowCmd_ IDL, so no new message type). If no
    // message arrives within the timeout the arms fall back to the carry pose.
    arm_sub_ = std::make_shared<unitree::robot::g1::subscription::LowCmd>("rt/arm_targets");
    arm_sub_->set_timeout_ms(200);
}

namespace {
// Pelvis pitch (+ forward) and roll in degrees from the measured orientation.
void pelvis_tilt_deg(LowState_t & ls, float & pitch, float & roll, float & side)
{
    const auto & q = ls.msg_.imu_state().quaternion();  // w, x, y, z
    const float w = q[0], x = q[1], y = q[2], z = q[3];
    const float gx = -2.0f * (x * z - w * y);
    const float gy = -2.0f * (y * z + w * x);
    const float gz = -(1.0f - 2.0f * (x * x + y * y));
    pitch = std::atan2(gx, -gz) * 57.2958f;
    roll = std::atan2(-gy, -gz) * 57.2958f;
    side = std::asin(std::clamp(std::fabs(gy), 0.0f, 1.0f)) * 57.2958f;  // sideways tilt, valid at any pitch
}
}  // namespace

void State_RLBase::begin_safety()
{
    const auto & jmap = env->robot->data.joint_ids_map;
    const int n_lower = env->action_manager->total_action_dim();
    q_entry_.resize(n_lower);
    for (int i = 0; i < n_lower; ++i)
        q_entry_[i] = lowstate->msg_.motor_state()[(int)jmap[i]].q();
    speed_ms_.assign(n_lower, 0.0f);
    enter_tp_ = speed_last_tp_ = std::chrono::steady_clock::now();
    policy_ready_ = false;
    tilt_violating_ = false;

    float pitch, roll, side;
    pelvis_tilt_deg(*lowstate, pitch, roll, side);
    start_refused_ = std::fabs(pitch) > start_max_tilt_deg_ || side > start_max_tilt_deg_;
    if (start_refused_)
        spdlog::error("Safety: policy start refused, robot is tilted (pitch {:+.0f} deg, roll {:+.0f} deg, "
                      "limit {:.0f}) -> Passive", pitch, roll, start_max_tilt_deg_);
}

bool State_RLBase::start_refused()
{
    const bool r = start_refused_;
    start_refused_ = false;
    return r;
}

bool State_RLBase::tilt_guard()
{
    float pitch, roll, side;
    pelvis_tilt_deg(*lowstate, pitch, roll, side);
    const float lean_deg = isaaclab::mdp::lean_state().value * 57.2958f;
    const float fwd_limit = std::max(tilt_fwd_deg_, lean_deg + tilt_lean_margin_deg_);
    const bool bad = side > tilt_roll_deg_ || pitch < -tilt_back_deg_ || pitch > fwd_limit;
    if (!bad) { tilt_violating_ = false; return false; }
    const auto now = std::chrono::steady_clock::now();
    if (!tilt_violating_) { tilt_violating_ = true; tilt_since_ = now; return false; }
    if (std::chrono::duration<float>(now - tilt_since_).count() < tilt_hold_s_) return false;
    spdlog::error("Safety: tilt pitch {:+.0f} deg (allowed {:+.0f}..{:+.0f}), sideways {:.0f} deg (allowed {:.0f}) "
                  "for {:.0f} ms -> Passive", pitch, -tilt_back_deg_, fwd_limit, side, tilt_roll_deg_,
                  1000.0f * tilt_hold_s_);
    tilt_violating_ = false;
    return true;
}

bool State_RLBase::speed_guard()
{
    // Running mean-square of each leg joint's speed. Walking stays near 2-5
    // rad/s RMS; a joint that rings or races goes well above 10.
    const auto now = std::chrono::steady_clock::now();
    const float dt = std::clamp(std::chrono::duration<float>(now - speed_last_tp_).count(), 0.0f, 0.02f);
    speed_last_tp_ = now;
    const auto & jmap = env->robot->data.joint_ids_map;
    const float a = dt / speed_tau_s_;
    for (size_t i = 0; i < speed_ms_.size(); ++i) {
        const float dq = lowstate->msg_.motor_state()[(int)jmap[i]].dq();
        speed_ms_[i] += a * (dq * dq - speed_ms_[i]);
        if (speed_ms_[i] > speed_rms_limit_ * speed_rms_limit_) {
            spdlog::error("Safety: leg joint {} speed RMS {:.1f} rad/s > {:.1f} -> Passive",
                          i, std::sqrt(speed_ms_[i]), speed_rms_limit_);
            speed_ms_[i] = 0.0f;
            return true;
        }
    }
    return false;
}

bool State_RLBase::action_runaway()
{
    const auto raw = env->action_manager->action();
    float peak = 0.0f;
    for (float v : raw) peak = std::max(peak, std::fabs(v));

    if (peak <= guard_max_abs_action_) {
        guard_violating_ = false;
        return false;
    }
    const auto now = std::chrono::steady_clock::now();
    if (!guard_violating_) {
        guard_violating_ = true;
        guard_since_ = now;
        return false;
    }
    if (std::chrono::duration<float>(now - guard_since_).count() < guard_hold_s_) {
        return false;
    }
    spdlog::error("RL guard: |raw action| {:.1f} > {:.1f} for {:.0f} ms -> Passive",
                  peak, guard_max_abs_action_, 1000.0f * guard_hold_s_);
    guard_violating_ = false;
    return true;
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    auto & jmap = env->robot->data.joint_ids_map;
    auto & qdef = env->robot->data.default_joint_pos;
    const int n = (int)jmap.size();
    const int n_lower = (int)action.size();
    const int n_arm = n - n_lower;

    // One-time init of the ramp buffer to the default (carry) arm pose.
    if (!arm_ramp_init_) {
        arm_q_applied_.assign(n_arm, 0.0f);
        for (int k = 0; k < n_arm; ++k) arm_q_applied_[k] = (float)qdef[n_lower + k];
        arm_ramp_init_ = true;
    }

    // run() is called at ~1 kHz (CtrlFSM), so use a wall-clock dt for the ramp.
    const auto now = std::chrono::steady_clock::now();
    const float dt = arm_first_run_
        ? (float)env->step_dt
        : std::chrono::duration<float>(now - arm_last_tp_).count();
    arm_last_tp_ = now;
    arm_first_run_ = false;
    // "Active" = an IL (GR00T) message arrived within the subscriber timeout.
    const bool arm_active = !arm_sub_->isTimeout();

    // Remote arm-pose selection via the gamepad (imported from arm.py). A live
    // rt/arm_targets (IL/GR00T) stream still overrides these when present.
    //   Y : toggle DEFAULT (carry) <-> FORWARD (arms out to receive a box)
    //   B : toggle FORWARD <-> BOX_HOLD (arms in, gripping) [only in those states]
    {
        auto js = env->robot->data.joystick;
        const bool y_now = js->Y.pressed;
        const bool b_now = js->B.pressed;
        if (y_now && !arm_prev_y_) {
            arm_pose_ = (arm_pose_ == ArmPose::DEFAULT) ? ArmPose::FORWARD
                                                        : ArmPose::DEFAULT;
        }
        if (b_now && !arm_prev_b_) {
            if      (arm_pose_ == ArmPose::FORWARD)  arm_pose_ = ArmPose::BOX_HOLD;
            else if (arm_pose_ == ArmPose::BOX_HOLD) arm_pose_ = ArmPose::FORWARD;
        }
        arm_prev_y_ = y_now;
        arm_prev_b_ = b_now;
    }

    // Arm-pose presets (imported from arm.py), mapped to the G1's 10 arm joints in
    // jmap arm order [L: sh_pitch, sh_roll, sh_yaw, elbow, wrist_roll, R: ...];
    // wrist_roll (idx 4/9) is 0. Applied only when the arm layout is the expected
    // 10 joints, else fall back to the carry/default pose.
    static const float POSE_FORWARD[10] = {
        -1.35f, 0.50f, 0.0f, 0.87f, 0.0f,  -1.35f, -0.50f, 0.0f, 0.87f, 0.0f};
    static const float POSE_BOXHOLD[10] = {
        -1.25f, 0.00f, 0.0f, 0.70f, 0.0f,  -1.25f,  0.00f, 0.0f, 0.70f, 0.0f};
    const bool pose_ok = (n_arm == 10);

    // Ramp rate: fast+transparent while tracking IL; gentle when homing to the
    // default/carry pose (finger-safe); moderate for a deliberate button pose.
    const float ARM_RATE_ACTIVE = 20.0f;  // rad/s, transparent tracking of IL
    const float ARM_RATE_RETURN = 0.7f;   // rad/s, gentle homing to default/carry
    const float ARM_RATE_POSE   = 2.0f;   // rad/s, deliberate button-selected pose
    float arm_rate;
    if (arm_active)                         arm_rate = ARM_RATE_ACTIVE;
    else if (arm_pose_ == ArmPose::DEFAULT) arm_rate = ARM_RATE_RETURN;
    else                                    arm_rate = ARM_RATE_POSE;
    const float max_step = arm_rate * dt;

    // Snapshot the IL arm command once (the DDS callback writes msg_ on another
    // thread).
    std::vector<float> il_q(n_arm), il_kp(n_arm), il_kd(n_arm), il_tau(n_arm);
    if (arm_active) {
        std::lock_guard<std::mutex> lock(arm_sub_->mutex_);
        for (int k = 0; k < n_arm; ++k) {
            const int motor = (int)jmap[n_lower + k];
            il_q[k]   = arm_sub_->msg_.motor_cmd()[motor].q();
            il_kp[k]  = arm_sub_->msg_.motor_cmd()[motor].kp();
            il_kd[k]  = arm_sub_->msg_.motor_cmd()[motor].kd();
            il_tau[k] = arm_sub_->msg_.motor_cmd()[motor].tau();
        }
    }

    for (int i = 0; i < n; ++i) {
        const int motor = (int)jmap[i];
        if (i < n_lower) {
            // Lower body: RL policy.
            // Soft start: blend from the pose at entry to the policy's target.
            // Before the first policy step the target is the entry pose itself.
            float alpha = 0.0f;
            if (policy_ready_) {
                const float t = std::chrono::duration<float>(now - enter_tp_).count();
                alpha = soft_start_s_ > 0.0f ? std::clamp(t / soft_start_s_, 0.0f, 1.0f) : 1.0f;
            }
            lowcmd->msg_.motor_cmd()[motor].q() = q_entry_[i] + alpha * (action[i] - q_entry_[i]);
            continue;
        }
        // Arm joint: follow IL target when live, else the button-selected pose
        // (FORWARD / BOX_HOLD), else the default carry pose.
        const int k = i - n_lower;
        float target;
        if (arm_active)                                     target = il_q[k];
        else if (pose_ok && arm_pose_ == ArmPose::FORWARD)  target = POSE_FORWARD[k];
        else if (pose_ok && arm_pose_ == ArmPose::BOX_HOLD) target = POSE_BOXHOLD[k];
        else                                                target = (float)qdef[i];
        const float err = target - arm_q_applied_[k];
        if (std::fabs(err) <= max_step) arm_q_applied_[k] = target;
        else arm_q_applied_[k] += std::copysign(max_step, err);

        lowcmd->msg_.motor_cmd()[motor].q() = arm_q_applied_[k];
        lowcmd->msg_.motor_cmd()[motor].dq() = 0;
        if (arm_active) {
            lowcmd->msg_.motor_cmd()[motor].kp() = il_kp[k];
            lowcmd->msg_.motor_cmd()[motor].kd() = il_kd[k];
            lowcmd->msg_.motor_cmd()[motor].tau() = il_tau[k];
        } else {
            lowcmd->msg_.motor_cmd()[motor].tau() = 0;
        }
    }
}