# TWIST-on-Nokov

First, please make sure you have configured GMR and TWIST environments.If not，visit [GMR](https://github.com/YanjieZe/GMR) and [TWSIT](https://github.com/YanjieZe/twist)

## Installation

```bash
cd GMR
git clone https://github.com/Steven-Uranus/TWIST-on-Nokov.git && cd TWIST-on-Nokov
mv nokov_vendor ../general_motion_retargeting/
mv XINGYING_sdk ../third_party/
mv nokov_to_twist.py ../scripts/
cd .. && rm -r TWIST-on-Nokov
```

## Usage

Enter Twist and start the lower-level controller.

```bash
cd TWIST/deploy_real && conda activate twist
python server_low_level_g1_sim.py --policy_path PATH/TO/YOUR/JIT/MODEL
```

Enter GMR and start the Nokov receiving server.
You need to make sure that your device is on the same IP address (or the same network) as the server so that you can receive data from it.

```bash
cd GMR && conda activate gmr
```
```bash
python scripts/nokov_to_twist.py --server_ip 192.168.110.xxx --robot_gmr unitree_g1 --human_height 1.6 --offset_to_ground --freq 50
```
