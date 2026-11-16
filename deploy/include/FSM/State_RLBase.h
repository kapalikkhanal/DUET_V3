// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <vector>
#include <chrono>
#include <atomic>

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        guard_violating_ = false;
        begin_safety();

        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            const int motor = static_cast<int>(env->robot->data.joint_ids_map[i]);
            lowcmd->msg_.motor_cmd()[motor].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[motor].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[motor].dq() = 0;
            lowcmd->msg_.motor_cmd()[motor].tau() = 0;
        }

        env->robot->update();
        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                env->step();
                policy_ready_ = true;

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    bool policy_thread_running = false;

    // Upper-body (arm) targets from the IL / GR00T manipulation policy. These are
    // published as a LowCmd on a SEPARATE topic ("rt/arm_targets") so they do not
    // collide with this controller's single rt/lowcmd writer. When the stream is
    // fresh the arm joints follow these targets; otherwise they hold the default
    // (carry) pose. A velocity-limited ramp smooths the carry <-> IL handoff.
    std::shared_ptr<unitree::robot::g1::subscription::LowCmd> arm_sub_;
    std::vector<float> arm_q_applied_;
    bool arm_ramp_init_ = false;
    bool arm_first_run_ = true;
    std::chrono::steady_clock::time_point arm_last_tp_;

    // Remote arm-pose state machine (gamepad Y/B), imported from arm.py:
    //   Y: DEFAULT <-> FORWARD ,  B: FORWARD <-> BOX_HOLD.
    enum class ArmPose { DEFAULT, FORWARD, BOX_HOLD };
    ArmPose arm_pose_ = ArmPose::DEFAULT;
    bool arm_prev_y_ = false;
    bool arm_prev_b_ = false;

    // Runaway-policy guard (see the constructor in State_RLBase.cpp). Defaults
    // are overridable from deploy.yaml `safety:`.
    bool action_runaway();
    float guard_max_abs_action_ = 15.0f;
    float guard_hold_s_ = 0.06f;
    bool guard_violating_ = false;
    std::chrono::steady_clock::time_point guard_since_;

    // Further guards, all -> Passive (damping), all overridable from
    // deploy.yaml `safety:` (see the constructor in State_RLBase.cpp).
    void begin_safety();   // called on entering the state
    bool tilt_guard();     // pelvis tilted beyond what the policy is trained for
    bool speed_guard();    // a leg joint is shaking or racing
    bool start_refused();  // the robot was not upright when the policy was started
    float tilt_roll_deg_ = 40.0f;         // sideways
    float tilt_back_deg_ = 35.0f;         // backward
    float tilt_fwd_deg_ = 50.0f;          // forward (deep squats lean ~30 deg)
    float tilt_lean_margin_deg_ = 25.0f;  // forward allowance beyond the commanded lean
    float tilt_hold_s_ = 0.10f;
    bool tilt_violating_ = false;
    std::chrono::steady_clock::time_point tilt_since_;
    float speed_rms_limit_ = 10.0f;       // rad/s, running RMS of each leg joint
    float speed_tau_s_ = 0.20f;           // averaging time of that RMS
    std::vector<float> speed_ms_;
    std::chrono::steady_clock::time_point speed_last_tp_;
    float start_max_tilt_deg_ = 20.0f;
    bool start_refused_ = false;
    // Soft start: the leg targets blend from the pose the robot is in when the
    // policy starts to the policy's own targets over soft_start_s_.
    float soft_start_s_ = 0.5f;
    std::vector<float> q_entry_;
    std::chrono::steady_clock::time_point enter_tp_;
    std::atomic<bool> policy_ready_{false};
};

REGISTER_FSM(State_RLBase)
