# Executive Summary  
This report details steps to deploy OpenVLA-OFT (a vision-language-action model) on Clemson’s Palmetto2 HPC cluster and connect it to a Flexiv Rizon 4 robot.  We cover Palmetto2 hardware (1,206 nodes, 54,540 CPU cores, 351,745 GB RAM, 1,615 GPUs), software environment (Rocky Linux 8, kernel drivers, CUDA/NVIDIA support, MPI), and access policies (Clemson accounts via ColdFront, SSH with Duo 2FA, no root access).  We discuss job scheduling under Slurm (default `work1` partition for batch jobs), GPU allocation (`--gres=gpu:N`), and storage (250 GB home, 10 TB scratch).  Next, we compare native vs containerized deployment (Docker vs Apptainer/Singularity) in a summary table.  We list prerequisites: compatible compilers (GCC, CUDA toolkit), Python (via `miniforge3/24.3.0` or `anaconda3/2023.09` modules), and libraries (PyTorch 2.x, TorchVision, Torchaudio, Flash-Attention, etc.).  We provide example Slurm scripts, module commands, and Dockerfile/Apptainer recipes for reproducible builds.  

For the Flexiv Rizon 4, we summarize the Flexiv RDK and ROS2 Bridge requirements: Debian/Ubuntu Linux (ROS 2 Humble/Jazzy) on x86_64, and a wired network with low latency.  We outline network setup: the robot’s control box has “User Port 1” (static IP 192.168.2.100) and “General Port” (DHCP server 192.168.100.1).  A cluster node must be configured (static IP or via DHCP/router) on the same subnet, and firewall rules must allow RDK traffic.  We include a network-topology diagram (below) showing the cluster-to-robot connection.  We give example ROS 2 launch commands (`flexiv_bringup`) that use the robot serial number, and discuss real-time constraints (prefer wired LAN, round-trip <1 ms for RT control).  For safety and security, we recommend limiting network exposure (e.g. VPN, secure VLAN, whitelists).

The report ends with validation (unit tests, LIBERO simulation vs real robot trials), logging/monitoring (Slurm `sacct`, Palmetto `jobperf`/`jobstats`, ROS logs), fault handling (Slurm checkpointing, RDK error callbacks), and performance tuning (batch sizing, mixed precision, GPU affinity).  An actionable checklist of pre- and post-deployment tasks is provided. The deployment plan assumes certain details (e.g. node GPU models, network topology) which should be verified. All instructions prioritize official sources (Clemson RCD docs, Flexiv manuals, OpenVLA-OFT docs, ROS2 docs).

## Palmetto2 Cluster Prerequisites  
- **OS and Kernel:** Palmetto2 runs Rocky Linux 8 (RHEL-compatible).  Ensure compute nodes have Rocky 8’s kernel and that security patches are current.  The default Python is 3.6 (EOL), so use a newer Python via modules: e.g. `module load miniforge3/24.3.0-0` (Python 3.10.14) or `anaconda3/2023.09-0` (Python 3.11.5).  
- **Compilers and MPI:** Load updated compilers (`gcc/12.3.0`, Intel or AOCC) and MPI (OpenMPI 4.x or newer).  For CUDA/GPU work, load the NVIDIA CUDA module, e.g. `module load cuda/12.3.0` (Palmetto provides optimized modules on NVIDIA nodes).  Install NVidia drivers and CUDA libraries (typically pre-installed). Ensure `nvidia-smi` works.  
- **GPU/CPU:** Palmetto2 has many GPU nodes (likely NVIDIA A100/H100, up to 80 GB VRAM).  OpenVLA-OFT inference needs ~16–18 GB (LIBERO/ALOHA) and training may need 27–80 GB per GPU.  Verify GPU model and memory on target nodes, and adjust job requirements.  CPU cores: select multi-core (e.g. `--cpus-per-task=4`) for data loading.  Ensure AMD/NVIDIA architecture modules are set appropriately (e.g. use `cuda` compiler module to access GPU-optimized builds).  
- **Network:** All nodes have at least 10/25 Gbps Ethernet; latest nodes also have InfiniBand (56–400 Gbps).  Use Ethernet for robot comms; ensure 10/25 Gbps NIC is used.  Configure cluster firewall or network ACLs to allow traffic (especially TCP ports used by Flexiv RDK/ROS; see Flexiv docs).  
- **Storage:** Users get 250 GB home and 10 TB scratch. Allocate sufficient space for datasets and model checkpoints.  Use Lustre scratch for large temporary data. File I/O at runtime can use `$TMPDIR` (fast local SSD) bound into jobs.  
- **Container Runtimes:** Docker is not available on compute nodes (no root).  Palmetto2 supports Apptainer (Singularity) containers.  We recommend using Apptainer images for reproducibility.  (Docker may be used locally to build images, then converted to `.sif` on cluster.)  
- **MPI/Distributed:** For multi-GPU training, use OpenMPI or SLURM’s native GPU parallelism (DDP).  Palmetto supports multi-node jobs with IB (`--mpi=openmpi`). Ensure `mpirun` is from loaded MPI module.  

```bash
# Example: Prepare environment on a Palmetto2 node
module load miniforge3/24.3.0-0 cuda/12.3.0 openmpi/5.0.1
conda create -n openvla-oft python=3.10  # create conda env for OpenVLA-OFT
conda activate openvla-oft
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 
pip install flash-attn==2.5.5 packaging ninja  # per OpenVLA-OFT SETUP
git clone https://github.com/moojink/openvla-oft.git
pip install -e openvla-oft
```

## User Access and Security  
- **Accounts:** Access is via Clemson University credentials.  Project leads must create ColdFront projects and add users. Approved allocations (General Queue) grant access to the `work1` partition within ~2 hours. All users must comply with Clemson IT policies.  
- **SSH:** Log in through `ssh username@slogin.palmetto.clemson.edu`. Clemson password and Duo 2FA are required. Verify the host fingerprint on first connect. No direct root or `sudo` access on compute nodes is provided (users run jobs as themselves). File permissions in $HOME follow standard Linux user settings (700 on home, 755 for dirs by default).  
- **Kerberos/LDAP:** Clemson uses centralized authentication (likely Kerberos/LDAP under the hood). Users should ensure they can log in to slogin with their Clemson login.  
- **File Security:** Private keys (for SSH) should be kept in `~/.ssh` with restrictive perms (600). Jobs should not expose secrets; use cluster vault or secure environment variables if needed. The cluster’s Acceptable Use guidelines forbid running interactive GUI or AI assistants on login nodes.  

## Job Scheduling & Resource Allocation  
- **Slurm Partitions:** Use `--partition=work1` for batch jobs. Interactive jobs (`srun --pty` or `sbatch --pty`) go to `interact` partition (auto-assigned). There are also owner-only partitions (for purchased nodes) and an OSG partition. By default, `sbatch` jobs run in `work1` with no preemption.  
- **SBATCH Options:** Allocate GPUs and CPUs via SBATCH directives. Example headers:  
  ```bash
  #!/bin/bash
  #SBATCH --job-name=openvla-finetune
  #SBATCH --partition=work1
  #SBATCH --gres=gpu:4           # request 4 GPUs
  #SBATCH --nodes=1             # or multiple if using DDP across nodes
  #SBATCH --ntasks=1 
  #SBATCH --cpus-per-task=8     # CPU threads for data loader
  #SBATCH --mem=100G            # RAM (adjust as needed)
  #SBATCH --time=48:00:00       # e.g. 2 days
  #SBATCH --output=job.%j.out
  #SBATCH --error=job.%j.err
  module load gcc/12.3.0 openmpi/5.0.1 cuda/12.3.0
  source activate openvla-oft
  srun python train_openvla_oft.py --config=config.yaml
  ```  
  In this example, `--gres=gpu:4` binds 4 GPUs to the job.  SLURM handles GPU allocation; PyTorch’s DDP will use these. Use `srun` (or `mpirun`) to launch MPI jobs. If binding specific GPU types is needed, use `--constraint` or `--comment` tags matching GPU models (see “Hardware Table” for constraints).  
- **GPU Binding:** Palmetto uses Slurm’s generic resource (gres) for GPUs. The job’s environment (e.g. `CUDA_VISIBLE_DEVICES`) is set automatically. For multi-GPU MPI jobs, use `--ntasks-per-node=<gpus-per-node>`.  Example for 2 nodes, 4 GPUs each: `#SBATCH --nodes=2 --gres=gpu:4 --ntasks-per-node=4`.  
- **QoS and Fair-Share:** The cluster enforces a 3,000-job submission limit. Fair-share may delay jobs if you are overusing resources. You can check priority with `sprio`. If jobs wait too long, consider smaller incremental runs or reserving nodes.  
- **Job Arrays:** For parameter sweeps, Slurm arrays are supported (`#SBATCH --array=0-9`). Each sub-job inherits the SBATCH resources.  
- **Open OnDemand:** Alternatively, use Clemson’s Open OnDemand portal for interactive development (launch GUI apps and shells on compute nodes). Jobs there are automatically placed in `interact`.  

## Containerization vs Native Install  
| **Option**        | **Pros**                                                      | **Cons**                                                        |
|-------------------|---------------------------------------------------------------|-----------------------------------------------------------------|
| Native (apt/conda)| Direct hardware access, potentially best performance; uses Palmetto modules (e.g. `cuda`, `gcc`). | Complex dependency management; changes affect user environment; harder to reproduce. |
| Docker (development) | Easy reproducibility; wide community images; supports `nvidia-docker`. | **Not allowed** on Palmetto (root required). Use only for building locally. |
| Apptainer (Singularity) | HPC-friendly container; runs as user without sudo; can import Docker images (`docker://rockylinux:8`); mounts `$HOME`. | Slight overhead vs native; read-only image (need rebuild to change). Must manage `.sif` files. |

1. *Native install:* We can `conda`/`pip` install directly on the node using provided modules.  Upgrading system libs (glibc, kernel modules) is not needed.  Use `conda create` with `miniforge3`.  Suitable for rapid testing but less reproducible.
2. *Docker:* We recommend *building* a Docker image (with CUDA base) locally, then convert it.  Example base image: `nvidia/cuda:12.3.0-cudnn8-ubuntu22.04`.  
   ```Dockerfile
   FROM nvidia/cuda:12.3.0-cudnn8-ubuntu22.04
   RUN apt-get update && apt-get install -y python3.10 python3-pip
   RUN python3.10 -m pip install --upgrade pip
   RUN python3.10 -m pip install torch==2.2.0 torchvision torchaudio \
       flash-attn==2.5.5 packaging ninja flexivrdk
   ```
   Build with `docker build -t openvla-ros:latest .`.  
   *Note: This Dockerfile is an example. On Palmetto2, Docker **cannot** run, so use this only to prepare an Apptainer image.*
3. *Apptainer (Singularity):* Use `apptainer build` or `pull`.  For example:
   ```bash
   # On a compute node, pull from Docker Hub:
   apptainer pull --name openvla.sif docker-daemon:openvla-ros:latest
   # OR directly from Docker registry (Ubuntu with ROS2):
   apptainer pull openvla-ros.sif docker://armadillo/ros2:humble
   ```
   Inside the container, one could run `ros2`, `flexivrdk`, and OpenVLA-OFT (after pip installing it in the container). Bind mounts for `$HOME` and `/scratch` let the container use shared storage.  

## OpenVLA-OFT Build & Dependencies  
- **Language & Libraries:** OpenVLA-OFT is Python-based and requires Python 3.10.  Key Python libs: PyTorch 2.x (CUDA build), TorchVision/Torchaudio (matched CUDA), [`flash-attn`](https://github.com/HazyResearch/flash-attention) 2.5.5, `packaging`, `ninja`.  We pip-install these in a fresh env. The repository itself is installed via `pip install -e`.  Ensure CUDA drivers match the PyTorch CUDA version.  
- **GPU Requirements:** As per OpenVLA-OFT docs, inference on ALOHA needs ~18 GB GPU RAM, training uses 27–80 GB.  If GPUs have less memory, use gradient accumulation or mixed precision.  For multi-GPU DDP training, no model sharding is needed; Slurm’s `srun` handles MPI.  
- **Compilers:** If building any C++ extensions (e.g. some PyTorch ops), load GCC 12.  Most of the work is Python, so we rely on binary wheels.  
- **CUDA/cuDNN:** Palmetto modules include CUDA Toolkit and cuDNN; just `module load cuda` is enough. Verify `nvcc --version`.  
- **MPI:** If running distributed PyTorch, use Slurm’s built-in launch (`srun`) or `torchrun --rdzv` as needed.  We may load `openmpi/5.0.1` (CUDA-aware MPI) to use MPI for NCCL backend.  
- **ROS (if used):** If the application calls ROS 2, install ROS 2 Humble or Jazzy.  Palmetto2 (Rocky) does not provide ROS, so use a container (Ubuntu). Flexiv provides ROS2 support via `flexiv_ros2` (Humble).  
- **Flexiv RDK (Robot Control Library):** Install the Flexiv RDK Python package on whichever system talks to the robot.  On a cluster node or container, run:  
  ```bash
  python3.10 -m pip install numpy spdlog flexivrdk  # Flexiv RDK (Python)
  ```  
  After installation, test by importing `flexivrdk` and creating `Robot("Your-Serial")`. A valid license (dongle or file) must be configured on the Rizon 4.  
- **Build Summary:** The key build steps (condensed) are:  
  1. `module load miniforge3/24.3.0-0 cuda/12.3.0 gcc/12.3.0 python3/3.10.14` (or activate conda env).  
  2. Create Python env: `conda create -n openvla-oft python=3.10`.  
  3. `source activate openvla-oft`.  
  4. `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117`.  
  5. `pip install flash-attn==2.5.5 packaging ninja`.  
  6. `git clone https://github.com/moojink/openvla-oft.git; cd openvla-oft`.  
  7. `pip install -e .`.  
  8. (Optional) Install `flexivrdk` and `flexiv_ros2` if controlling the robot via ROS.  

## Device Connectivity (Cluster ↔ Rizon 4)  

```mermaid
graph LR
  Login[SSH Login Node] --> |SSH| Compute[Compute Node (GPU)]
  Compute --> |Ethernet| CampusNet[Campus Network]
  CampusNet --> |Wired LAN| Switch
  Switch --> RobotPort[Flexiv Rizon 4 Control Box]
  RobotPort --> Rizon[Flexiv Rizon 4 Robot]
```

- **Network Setup:** The Rizon control box has three Ethernet ports; typically use “User Port 1” for direct control.  By default: **User Port 1** is static `192.168.2.100`, **User Port 2** is DHCP, **General Port** is a DHCP server at `192.168.100.1`.  For example, to connect from a cluster node: plug that node’s NIC into User Port 1 and assign it a static IP like `192.168.2.101`/24 (netmask 255.255.255.0).  Alternatively, if on the same building network, connect both to a switch/router and let DHCP assign different 192.168.2.x addresses.  
- **MTU and Firewall:** Set the node’s NIC MTU to 1500 to match the robot (especially on direct link).  Disable any host firewalls or explicitly allow the RDK/ROS ports.  Flexiv RDK uses TCP and UDP (see manual); at minimum, the robot server port (default 8080) and RDK heartbeat must pass.  One can temporarily disable iptables/UFW on the node, or whitelist the `flexivrdk` binary.  
- **Latency:** For **real-time control**, use wired LAN (User Port 1) with round-trip latency <1 ms. For non-real-time/ROS operation, wireless is permitted but still keep latency <1000 ms. Measure with `ping <robot_ip>` (e.g. `ping 192.168.2.100`) and ensure ~0.x ms. High-latency (>1 ms) or packet loss can cause missed deadlines.  
- **ROS Communication:** If using ROS 2, ensure both cluster node and robot are on same ROS domain (set `ROS_DOMAIN_ID`). Launch the `flexiv_ros2` bridge on a system with ROS2 Humble. The ROS bridge will advertise topics (joint states, commands, gripper control). Adjust `/etc/hosts` or DNS so `RobotHostName` resolves.  
- **Security:** It’s safest to isolate the robot network (e.g. private VLAN). If remote access is needed (off-campus), use a VPN into Clemson’s network or SSH tunnels. Do **not** expose the robot ports to the Internet. For example, one can forward port 8080 from a secure jump host to the cluster node.  
- **Hardware Access:** The cluster cannot directly control robot hardware except via network. No USB or serial links from HPC to robot are expected. The Flexiv tablet UI and emergency stop remain on-site. All commands flow through the RDK over Ethernet.  

## Flexiv Rizon 4 and ROS Integration  
- **Flexiv RDK:** After network is set up, test RDK connectivity on a compute node or container:  
  ```python
  from flexivrdk import Robot
  robot = Robot("Enlight-L-<serial>")  # use actual serial without spaces
  print(robot.getSerialNumber())
  ```  
  This should detect the robot. If not, check firewall and IP. The RDK python package supports both real-time RT modes and NRT modes (here we likely use RT modes). Install RDK on the same node or container as OpenVLA.  
- **ROS 2 Bridge:** Flexiv provides `flexiv_ros2` (Ubuntu Humble) packages. Build and source it as per docs. Key launch examples:  
  ```bash
  source /opt/ros/humble/setup.bash
  source ~/flexiv_ros2_ws/install/setup.bash
  ros2 launch flexiv_bringup flexiv.launch.py robot_sn:=<serial> robot_type:=Enlight-L
  ```  
  This connects to the robot server and starts the default joint-position controller. To use joint impedance mode, add `rdk_control_mode:=joint_impedance`. MoveIt demos are also available once the hardware interface is up.  
- **Real-Time Constraints:** For high-rate control loops, consider running on a node with deterministic performance. Palmetto nodes are not real-time, but RDK can still operate in non-RT mode (<1000 ms latency) for planning. The Flexiv RDK supports real-time kernel on Ubuntu (optional) but for our case we’ll run on stock Palmetto OS and accept non-RT. Ensure ROS 2 executors don’t overload the CPU.  
- **Firewall/NAT/VPN:** If cluster cannot reach the robot IP due to network segmentation, either (a) place the robot on the campus network accessible to Palmetto, or (b) use Clemson’s VPN into the internal network.  Consult Clemson network admins if needed.  

## Testing, Validation, and Monitoring  
- **Unit Tests:** Verify OpenVLA-OFT installation by running included demos on LIBERO data. Use sample observations to generate an action chunk (see [73†L254-L263]). Ensure no errors on a compute node GPU.  
- **Simulation vs Hardware:** First run policies in simulation (e.g. with Gazebo or Flexiv’s demo modes) to validate end-to-end control. Flexiv’s RDK includes a “simulated robot” mode (no real hardware). Then test on the actual Rizon: execute safe motions (no-load, guarded speed) before running full tasks. Include a “dry-run” that checks motion plan without moving (e.g. MoveIt’s plan-only).  
- **ROS Launch Testing:** Use a ROS launch file (e.g. `flexiv_moveit_config/demo.launch.py`) to command simple motions. Log ROS topics (`rostopic echo`) to confirm sensor data flows.  
- **Slurm Job Tests:** Submit a short SBATCH job that runs inference on a small batch of observations. Check Slurm logs (`sbatch --mail-type=ALL`) and `sacct` for resource usage.  
- **Monitoring:** Use Palmetto’s tools: `jobperf <jobid>` or `jobstats` to check CPU/GPU utilization. Within ROS, use `ros2 topic list` and `ros2 topic hz` to monitor rates. Record `robotState` and `clock` in ROS for later analysis.  
- **Logging:** Capture stdout/stderr in files. Enable ROS logging (e.g. `ros2 launch ... --log-level debug`). Save RDK logs (it logs to console by default).  
- **Fault Recovery:** If a Slurm job crashes, use its error output and possibly restart. Use ROS safety calls if robot fault occurs (e.g. RDK’s fault handlers). For network drops, RDK can auto-stop the robot on lost connection.  
- **Performance Tuning:** Profile GPU memory and bandwidth. Adjust OpenVLA batch size for available GPU RAM. Consider mixed precision (bfloat16) and PyTorch’s `torch.cuda.amp` to increase throughput. On multi-GPU runs, ensure PCIe/NVLink usage is balanced.  

## Diagrams  
**Network Topology:** The mermaid diagram above shows a typical setup: the user SSHs into a Palmetto login node, moves to a GPU compute node, which connects over Clemson’s LAN (or direct link) to the Rizon’s control box and robot. All control traffic (RDK, ROS topics) flows via Ethernet.  

**Deployment Timeline:** The following flowchart outlines major steps. Replace `<...>` with specifics as needed.  

```mermaid
flowchart LR
    A[Gather requirements & specs] --> B[Configure Palmetto2 access (ColdFront, modules)]
    B --> C[Prepare environment: install PyTorch, OpenVLA-OFT, Flexiv RDK]
    C --> D[Connect cluster to Rizon network (static IPs, firewall)]
    D --> E[Build/launch containers or native env]
    E --> F[Deploy ROS 2 bridge and OpenVLA-OFT on node]
    F --> G[Run sample jobs in simulation/LIBERO]
    G --> H[Test on Rizon 4 (basic moves)]
    H --> I[Full integration testing (task runs)]
    I --> J[Monitoring/log collection]
    J --> K[Optimize & tune performance]
    K --> L[Document and finalize deployment]
```

## Pre-deployment Checklist  
- [ ] **Confirm Palmetto2 specs:** GPU models (check `sinfo -Nl`) and available memory. Verify node OS/kernel (Rocky 8), CUDA version.  
- [ ] **ColdFront project & account:** Ensure user is on an approved allocation (General Queue). Confirm `srun` can allocate GPUs.  
- [ ] **Install prerequisites on cluster:** Decide on container vs native. If native, load modules (CUDA, Python). If container, build `.sif` with CUDA and Python.  
- [ ] **Python env:** Create and test conda env for OpenVLA-OFT on a compute node (use one GPU). Confirm `python -c "import torch"` sees GPU.  
- [ ] **OpenVLA-OFT code:** Clone repo, install as editable. Run a quick inference script (e.g. [73†L254-L263]) to verify setup.  
- [ ] **Flexiv RDK:** Obtain RDK library and license. Install `flexivrdk` on the node or container. Test robot connectivity (ping and RDK).  
- [ ] **Network plan:** Decide which LAN port on Rizon to use. Reserve IP range. Coordinate with network admin if placing on campus net.  
- [ ] **Firewall rules:** If needed, open ports 8080 (RDK) and any additional (e.g. 5300 UDP for RDK heartbeat). Test SSH (if using tunnel).  
- [ ] **ROS 2 environment:** If using ROS, prepare an Ubuntu ROS container or separate VM. Clone and build `flexiv_ros2`.  
- [ ] **Slurm scripts:** Draft job scripts for training and inference. Test with a “sleep” job or small workload.  

## Post-deployment Checklist  
- [ ] **Connectivity:** Verify end-to-end: Palmetto node → Flexiv RDK → Rizon (e.g. get serial number).  
- [ ] **Basic motion test:** Run a safe motion (e.g. move arm to ready pose) via RDK/ROS, with no payload. Confirm sensing.  
- [ ] **Data logging:** Ensure all critical logs (ROS bags, model output, Slurm logs) are being saved to persistent storage.  
- [ ] **Job monitoring:** Monitor resource use with `jobperf`/`jobstats`. Adjust SBATCH resources if needed (e.g. add `--mem` or CPUs).  
- [ ] **Performance:** Check GPU utilization during runs (`nvidia-smi`, PyTorch profiler). Tune batch size or concurrency.  
- [ ] **Error handling:** Simulate common faults (network disconnect, large load) and ensure safe shutdown or reconnection as per RDK guidelines.  
- [ ] **Documentation:** Record final configuration (module versions, container SHA, network settings) and store scripts in version control.  

**Sources:** Clemson Palmetto2 docs, Flexiv RDK manual, and OpenVLA-OFT repository/docs provide the above specifications and procedures.