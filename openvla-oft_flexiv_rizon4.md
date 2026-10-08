# Setup OpenVLA-OFT on Flexiv Rizon 4

## Executive Summary  
This report details the full procedure to integrate the OpenVLA-OFT vision-language-action model with a Flexiv Rizon 4 robot.  It covers hardware/firmware requirements, OS and software dependencies, networking and real-time configuration, step-by-step installation, model deployment, testing, troubleshooting, performance tuning, and safety considerations.  The Rizon 4 uses a HESPER controller (TCP/IP, Profinet, Modbus interfaces); it is 7-DOF, 4 kg payload, 876 mm reach, IP65-rated and CE/ETL-certified.  We assume Flexiv Elements (robot OS) v3E.1 or later (compatible with RDK v2.1), Ubuntu 22.04 with Python 3.10 on the PC, and an NVIDIA GPU (≥18 GB).  Key steps: enable RDK remote mode and license via the teach pendant; install Flexiv RDK (Python API) and OpenVLA-OFT (Python/conda); connect PC↔robot via Ethernet (static IP or DHCP); run the OpenVLA-OFT inference loop to produce joint commands sent to the robot.  Critical safety checks and troubleshooting (connection, faults) are included.  

```mermaid
flowchart LR
  Robot["Flexiv Rizon 4 Robot\n(7-DOF, HESPER controller)"] 
  PC["Control PC\n(Ubuntu Linux, Python 3.10)"]
  Camera["Camera\n(Optional wrist or external)"]
  PC -->|Ethernet (TCP/IP)| Robot
  Camera -->|Images| PC
  PC -->|Flexiv RDK (Python API)| Robot
  PC -->|OpenVLA-OFT model| Robot
```

## Required Hardware and Firmware  
- **Robot and Controller:** Flexiv Rizon 4 (7 DOF, 4 kg payload, 876 mm reach, IP65).  The HESPER controller (16 DI/DO, 500 W, 12 kg) supports Profinet/Modbus/TCP/IP communications.  The robot requires Flexiv Elements software v3E.1 (Jun 2026) for compatibility with RDK v2.1.  A teach pendant and motion bar (with emergency stop) are used for enabling and mode switching.  
- **RDK License:** A Flexiv RDK license (Standard or Professional) must be installed via the UI.  This enables *Remote Mode* for API control.  
- **PC:** A workstation running Ubuntu 22.04 LTS (64-bit) with Python 3.10.  Multi-core CPU (≥4 cores), ≥32 GB RAM, and an NVIDIA GPU with ≥16–18 GB VRAM are recommended.  Example GPUs: RTX 4090 or A100.  If using Ubuntu 20.04, apply PREEMPT_RT patches or use Ubuntu Pro for a real-time kernel.  
- **Camera:** For vision input, a compatible camera (e.g. Flexiv wrist camera or USB camera) should be mounted.  Ensure camera drivers and calibration are set up on the PC.  
- **Accessories:** Optional Force/Torque (F/T) sensor or external axes as needed (Rizon 4 supports these options).  All mounting, cabling and safety fixtures must follow Flexiv guidelines.  

| Component             | Requirement / Version              | Notes                                    |
|-----------------------|------------------------------------|------------------------------------------|
| Flexiv Robot (Rizon 4)| 7 DOF, 4 kg payload, IP65      | Safe (CE/ETL-certified)      |
| Robot Controller      | HESPER (Ethernet, Profinet, Modbus) | Includes motion bar and E-stop hardware |
| Robot Software (Elements) | v3E.1 (Jun 2026)               | Compatible with Flexiv RDK v2.1 |
| Flexiv RDK (API)      | v2.1 (Python/C++)                   | Install via pip or source   |
| PC OS                 | Ubuntu 22.04 (or 20.04 w/ RT patch) | Real-time kernel optional     |
| Python                | 3.10                                 | Use Conda for environment isolation |
| PyTorch/CUDA          | PyTorch 2.2.0 (CUDA 12.x)   | torchvision 0.17.0, flash-attn 2.5.5    |
| GPU                   | NVIDIA (≥16 GB VRAM)  | Required for OpenVLA inference          |
| OpenVLA-OFT Code      | moojink/openvla-oft (latest)         | `git clone` + `pip install -e .` |
| Network               | 1 Gbps Ethernet                     | PC and robot on same LAN                 |
| Additional SDKs       | (Optional) Flexiv ROS2 bridge        | If integrating into a ROS2 system       |

## OS and Software Prerequisites  
- **Operating System:** Ubuntu 22.04 LTS (supports real-time kernel via Ubuntu Pro).  Windows 10/11 or macOS are supported by RDK but are not recommended for heavy ML workloads.  
- **Python Environment:** Use Python 3.10 (Conda environment recommended).  The RDK Python API and OpenVLA-OFT are verified on Python 3.10.  
- **Flexiv RDK:** Install Flexiv RDK v2.x (Python) via pip.  On the PC:  
  ```bash
  pip3 install numpy spdlog flexivrdk
  ```  
  This installs the Flexiv API bindings.  C++ development requires CMake 3.22+ and a C++ compiler (not detailed here).  
- **CUDA and GPU Drivers:** Install NVIDIA GPU drivers and CUDA Toolkit matching PyTorch CUDA version (e.g. CUDA 12.4 for PyTorch 2.2.0). Verify GPU is recognized.  
- **PyTorch & ML Libraries:** In the Conda env:  
  ```bash
  conda create -n openvla-oft python=3.10 -y
  conda activate openvla-oft
  pip install torch torchvision torchaudio  # see PyTorch site for appropriate CUDA build
  ```  
  Use the PyTorch stable (2.2.0) and TorchVision 0.17 as pinned in OpenVLA’s requirements. Also install HuggingFace Transformers (pulls via `pip install -e .` below).  
- **OpenVLA-OFT Dependencies:** Clone the OpenVLA-OFT repo and install:  
  ```bash
  git clone https://github.com/moojink/openvla-oft.git
  cd openvla-oft
  pip install -e .
  ```  
  This pulls in required packages (e.g. `transformers`, `accelerate`, `datasets`).  Optionally install FlashAttention 2 for faster training (`pip install "flash-attn==2.5.5"`) and any extras like `Packaging`, `ninja`.  
- **Other Tools:** For network config, ensure `iproute2` (for `ip link`) and `nmtui` (if needed) are available (usually pre-installed).  

## Drivers, SDKs, and Middleware  
- **Flexiv RDK (Robotic Development Kit):** Provides low-level real-time (RT) and high-level non-real-time (NRT) APIs.  The Python package `flexivrdk` exposes `Robot`, `RobotStates`, `RobotActions`, etc.  C++ SDK is available via source (requires building Flexiv’s library).  
- **Flexiv ROS2 Bridge (optional):** `flexiv_ros2` bridges RDK to ROS2 topics if a ROS-based system is needed. It depends on `rclcpp`/`rclpy`.  
- **NVIDIA CUDA/cuDNN:** For GPU acceleration. Must match the installed PyTorch build.  
- **OpenVLA-OFT (ML Stack):** Built on HuggingFace/Accelerate. It uses PyTorch 2.2 and Transformers 4.40. Ensure compatibility (see pinned versions in OpenVLA repo).  
- **Middleware:** Standard network stack (TCP/IP). No special middleware is required beyond the above.  For distributed setups, SSH and containerization (Docker) may be used but are optional.  

| Name                | Version/Requirement               | Role                                      |
|---------------------|------------------------------------|-------------------------------------------|
| Python (Conda)      | 3.10                              | Base language for RDK and OpenVLA-OFT     |
| Flexiv RDK (API)    | v2.x (Python)         | Robot control (RT/NRT commands)           |
| CUDA Toolkit        | ≥12.4                             | Required by PyTorch 2.2                   |
| NVIDIA Driver       | Matched to CUDA                   | GPU operations                            |
| PyTorch             | 2.2.0               | ML framework (OpenVLA inference)          |
| HuggingFace Transf. | 4.40.1              | Model tokenization and pipelines           |
| flash-attn          | 2.5.5               | (Optional) Fast attention layers          |
| OpenVLA-OFT repo    | Latest commit                     | VL-action model code (pip-installed)      |
| Flexiv ROS2 Bridge  | (if using ROS2)                   | ROS integration (optional)                |

## Network, Real-Time, and Security Configuration  
- **Ethernet Connection:** The Rizon 4 control box has three user Ethernet ports (User Port 1, User Port 2, General Port) with default IP modes:  
  - *User Port 1:* Static IP `192.168.2.100`, netmask `255.255.255.0`.  To use this port, set the PC to a static IP in `192.168.2.x` (e.g. `192.168.2.10/24`).  
  - *User Port 2:* DHCP (connect PC and robot through a common switch with DHCP).  
  - *General Port:* Robot acts as DHCP server (`192.168.100.1`).  Use automatic IP on PC.  
- **MTU:** For a direct PC↔robot link (User Port 1 or General Port), set MTU=1500 on the PC interface.  E.g.: `sudo ip link set dev eth0 mtu 1500`.  
- **Firewall:** Disable or whitelist RDK applications on the PC to allow UDP/TCP traffic.  If using Ubuntu’s UFW, e.g. `sudo ufw disable` for testing.  
- **Latency:** Ping the robot (e.g. `ping 192.168.2.100`) and ensure round-trip latency is <5 ms.  Network jitter impacts real-time control.  
- **Real-Time OS (optional):** For hard real-time control (sub-2 ms loop), use Ubuntu 22.04/24.04 with PREEMPT_RT (via Ubuntu Pro subscription).  Otherwise generic kernel yields ~4–10 ms loop time, sufficient for many applications.  
- **Security:** Keep the robot network isolated or secure. Install RDK license locally (no cloud needed). Use SSH or VPN if remote access is required. Ensure the emergency stop and mode switches on the teach pendant are functional. 

## Installation Steps  
1. **Prepare the Robot:** Mount the Rizon 4 on a rigid base. Connect power (use Flexiv-supplied cables) and turn on the robot.  Connect the teach pendant and motion bar.  
2. **Enable RDK Remote Mode:** On the teach pendant (Flexiv Elements UI), go to *Settings → Remote Mode*, select *Ethernet*, and reboot the robot. Install your RDK license under *Settings → License*. Switch the mode to *Auto (Remote)* via the motion bar or UI. This allows the PC to control the robot.  
3. **Connect PC to Robot:** Use an Ethernet cable to PC. For User Port 1, set PC IP static to `192.168.2.x/24` (e.g. `192.168.2.10`), gateway empty. Confirm link: `ping 192.168.2.100` should reply.  
4. **Install Flexiv RDK (Python):** On the PC, install required tools:  
   ```bash
   sudo apt update && sudo apt install -y python3-pip build-essential cmake
   pip3 install numpy spdlog flexivrdk
   ```  
   This adds the `flexivrdk` package. Test with a simple script that imports `flexivrdk` and connects:  
   ```python
   import flexivrdk
   robot = flexivrdk.Robot("Enlight-L-<Serial>")  # match your robot’s serial
   print("Connected:", robot.robot_states()["robot-state"])
   ```  
   If this fails, check Remote mode and firewall.  
5. **Install OpenVLA-OFT Environment:**  
   ```bash
   conda create -n openvla-oft python=3.10 -y
   conda activate openvla-oft
   pip install torch torchvision torchaudio      # or use specific CUDA as per PyTorch site
   git clone https://github.com/moojink/openvla-oft.git
   cd openvla-oft
   pip install -e .
   ```  
   This follows the OpenVLA-OFT setup. Ensure PyTorch is 2.2.0 (CUDA 12.x) and install FlashAttention if needed (`pip install "flash-attn==2.5.5"`).  
6. **Acquire Models and Data:** Download or prepare the OpenVLA-OFT checkpoint (e.g. from Hugging Face). For example, the pretrained LIBERO policy `moojink/openvla-7b-oft-finetuned-libero-spatial` is used in demos. Place any calibration data or vocab files in the working directory.  
7. **Verify Robot Control:** Run a built-in RDK example to ensure communication (e.g. `basics1_display_robot_states` in RDK examples). The robot should send periodic state updates at 1 kHz.  
8. **Run OpenVLA-OFT Inference:** Using the setup in step 5, run a test Python script (similar to the sample in OpenVLA-OFT README) to load the model, process a sample observation (images + state), and print the output action chunk. Ensure CUDA is visible.  

## Building and Deploying OpenVLA-OFT Components  
- **Inference Script:** Write a Python script that ties everything together. Key elements:  
  1. **Connect to Robot:**  
     ```python
     import flexivrdk
     from flexivrdk import Mode
     robot = flexivrdk.Robot("Enlight-L-<Serial>")
     robot.switch_mode(Mode.RT_JOINT_IMPEDANCE)  # or RT_CARTESIAN_MOTION_FORCE
     ```  
  2. **Capture Observation:** Acquire images (e.g. using OpenCV from the camera) and robot state (e.g. `robot.robot_states()["joint"]`). Construct an `observation` dict:  
     ```python
     observation = {
       "full_image": full_image,       # PIL Image or ndarray
       "wrist_image": wrist_image,     # if available
       "state": robot_state,           # e.g. 7-joint angles/velocities
       "task_description": "Place the red block in the box."
     }
     ```  
  3. **Load Model:** Use OpenVLA-OFT utilities to get the model and processor:  
     ```python
     from openvla_utils import get_vla, get_processor, get_action_head, get_proprio_projector, get_vla_action
     cfg = GenerateConfig(pretrained_checkpoint="moojink/openvla-7b-oft-finetuned-libero-spatial", 
                          use_proprio=True, num_images_in_input=2)
     vla = get_vla(cfg)
     processor = get_processor(cfg)
     action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
     proprio_proj = get_proprio_projector(cfg, llm_dim=vla.llm_dim)
     ```  
  4. **Run Inference:**  
     ```python
     actions = get_vla_action(cfg, vla, processor, observation, observation["task_description"],
                              action_head, proprio_proj)
     ```  
     This yields a sequence of continuous action vectors (e.g. joint velocity or position deltas).  Example usage is shown in OpenVLA-OFT docs.  
  5. **Send Commands to Robot:** Interpret the output (e.g. map to joint positions) and command the robot.  The example usage in OpenVLA shows `robot.act(action)`.  Internally, one can do:  
     ```python
     # Example: stream joint position commands at 1 kHz
     for target in actions:
         robot.stream_joint_position({"JointGroup0": target})
     ```  
     or use `robot.send_joint_position()` in NRT mode for discrete moves. Ensure the command matches the current control mode.  
- **Deployment Options:** You can run the above script directly on the PC connected to the robot. Alternatively, host the OpenVLA model on a separate server and communicate via a REST API (OpenVLA provides a sample server script). In either case, synchronize the control loop (~5–10 Hz) with robot feedback.  
- **Configuration:** Store configurations (e.g. model checkpoint name, image preprocessing) in a separate file or command-line args. Use version control for your code and note software versions.  

```mermaid
flowchart TD
  A[1. Setup Robot (mount, power on)]
  B[2. Enable RDK Server (license, remote mode)]
  C[3. Switch to Auto (Remote) mode]
  D[4. Connect PC: set static IP 192.168.2.x/24, MTU=1500]
  E[5. Verify Network (ping robot IP)]
  F[6. Install Flexiv RDK (Python): pip install flexivrdk]
  G[7. Install OpenVLA-OFT Env: conda env, pip install torch & openvla-oft]
  H[8. Download/Load Model Checkpoint]
  I[9. Run Example Inference & Control Loop]
  A --> B --> C --> D --> E --> F --> G --> H --> I
```

## Testing and Validation  
- **Basic Checks:** After setup, run RDK example programs (C++ or Python) to verify robot control and data flow (e.g. display joint states). Check `robot.operational()` returns true before commanding motion.  
- **Sensor Verification:** If using camera vision, capture test images and ensure they are correctly processed by the model’s image encoder.  
- **Model Output:** Test the OpenVLA-OFT script on a sample observation to ensure it produces reasonable joint commands (not NaNs or out-of-range).  Log the output for analysis.  
- **Dry Runs:** With the robot in a safe environment, command simple motions (e.g. move to a neutral pose) via the script. Verify the robot moves as expected. Gradually test more complex tasks.  
- **Integration Tests:** Evaluate the end-to-end pipeline on a known task (e.g. pick-and-place) and measure success rate. Compare against any ground-truth or expectations.  
- **Safety Checks:** Always begin with small movements and the E-Stop at hand. Monitor for unexpected behavior. Use `Robot.freeDrive()` mode (if needed) to manually guide the arm for calibration.  
- **Unit Tests:** Where possible, test individual components (network ping, `robot.robot_states()`, model inference on dummy data). Verify that faults are reported by the API and logged.  

## Common Failure Modes and Troubleshooting  
- **Connection Failure:** If `flexivrdk.Robot` instantiation fails, check that Remote mode is enabled on the robot and the correct serial number is used. Ensure the Ethernet cable is plugged into the correct port and that IP settings on the PC match the robot’s network mode. Disable any firewall blocking the RDK (or whitelist the RDK program).  
- **Version Mismatch:** Ensure the RDK Python version matches the robot software (see compatibility table). Update the robot software or use the correct RDK if needed.  
- **Licensing Errors:** If remote commands are rejected, verify the RDK license is installed and valid. The teach pendant UI should show the license status.  
- **Command Ignored:** If the robot does not move after sending actions, check the control mode: e.g. use `RT_JOINT_IMPEDANCE` for streaming commands or `NRT_JOINT_POSITION` for discrete sends. Use `robot.SwitchMode()` to change modes.  
- **Faults (Critical or Minor):**  
  - A **Critical Fault (CAT0)** (triggered by violating safety limits) will immediately stop the robot. If this occurs, call `robot.ClearFault()` (takes ~30 s), or power-cycle the robot if necessary. Ensure commands are within joint/force limits.  
  - **Minor Faults** (e.g. command errors) do not cut power and can be cleared quickly with `robot.ClearFault()`.  
- **Model-Related Issues:** GPU out-of-memory or CUDA errors indicate insufficient hardware; try reducing batch size or using `torch.cuda.empty_cache()`.  If inference is too slow, check that the model is loaded on GPU and not being reloaded each iteration.  
- **High Latency:** If control is sluggish, ensure the loop runs at ~5–10 Hz. OpenVLA is not trained on high-frequency (>10 Hz) action chunks; downsample commands to ~5 Hz if needed. Measure total latency from image capture to robot command.  
- **Coordinate Misalignment:** If the robot moves incorrectly relative to the camera frame, verify camera-to-robot extrinsics. Ensure the model’s frame of reference matches the robot’s (you may need to adjust the task description or calibration).  
- **Data Errors:** If the input images or state vectors have incorrect format/dimensions, the model will error. Follow the `GenerateConfig` and observation format exactly as in the OpenVLA-OFT examples.  
- **Debugging:** Use `robot.event_log()` and `robot.fault()` to retrieve error messages from the robot. Add verbose logging around the inference and command steps to isolate failures.  

## Performance Tuning and Latency Optimization  
- **Kernel and Scheduling:** Enable a real-time Linux kernel (Ubuntu Pro) to minimize OS-induced jitter. Use `sudo chrt` or thread priority APIs to elevate the Python process if necessary.  
- **Inference Optimizations:** For faster model inference, consider:  
  - **Quantization:** Load model in 8-bit or 4-bit mode (if supported by OpenVLA-OFT) to reduce memory and speed up computation.  
  - **TensorRT:** Convert the model to TensorRT for deployment on NVIDIA hardware (community tools are available) to achieve low-latency inference. NVIDIA documentation suggests converting directly on the target hardware.  
  - **Batching and Pipelining:** If processing multiple images or tasks, use batch inference. Keep the model loaded and avoid Python overhead in the real-time loop.  
- **Control Frequency:** Because OpenVLA was trained on ~5–10 Hz data, do not command the robot faster than this; generate action chunks of length N (e.g. 5–10 steps) and apply them sequentially.  
- **Network Tuning:** Use a high-quality Ethernet cable/switch to avoid packet loss. Confirm MTU is consistent (1500).  
- **Profiling:** Profile the inference loop (e.g. with Python’s `time` or `nvprof`) to find bottlenecks (image preprocessing vs model forward pass). Adjust the code accordingly (e.g. move tensor transforms to GPU).  

## Safety and Compliance Considerations  
- **Follow Official Safety Documentation:** As Flexiv stresses, “read through all documents shipped with the robot… and strictly follow all safety instructions”.  
- **Emergency Stop:** The Rizon 4 has an integrated E-Stop (motion bar) that cuts power on any critical fault. Test this in your setup to ensure it halts the robot immediately.  
- **Operating Conditions:** Ensure the robot’s ambient requirements (0–45 °C, 20–80% humidity) are met. Do not exceed the rated payload or speed.  
- **Software Licensing:** The OpenVLA models use Llama 2 under its license; for commercial use, ensure compliance with Meta’s terms.  
- **Physical Safety:** Keep a clear area around the robot. Do not place limbs or objects near the manipulator when running the system. Use safeguards (light curtains, safety mats) in compliance with local regulations (e.g. ISO 10218/ISO/TS 15066 for collaborative robots).  
- **Testing Safety:** Perform initial tests at reduced power/speed (using `SetOperationalMode` or limiting `max_lin_vel`) to prevent accidents. Verify the “Free Drive” mode for safe guiding if needed.  
- **Data Safety:** If camera images or logs contain sensitive info, handle them per your data policy.  

## References  
Key specifications, software versions, and procedures are drawn from official Flexiv documentation and OpenVLA resources.  For example, Rizon 4’s specs and HESPER interface are documented on Flexiv’s site.  Flexiv RDK manuals provide setup instructions and troubleshooting.  The OpenVLA-OFT GitHub and related papers describe model requirements and usage. All information above is cited from primary sources.