# AI-driven energy- and EMF-aware 5G RAN management with a Digital Twin

Research prototype (simplified, synthetic data) exploring three themes of 5G network management:
network performance (QoS), **energy efficiency** and **electromagnetic exposure**, using a
**Digital Twin** and **federated learning**.

## What it does
1. **Simulator** - 7 macro cells, 3.5 GHz / 100 MHz, log-distance path loss + shadowing, SINR,
   per-UE throughput, EARTH-style base-station power model, area-averaged EM-exposure proxy.
2. **Digital Twin** - gradient-boosting surrogate of the simulator. It only sees what an operator observes
   (per-cell traffic + configuration) and predicts energy, QoS and exposure.
3. **Optimisation** - the twin searches per-cell Tx power / sleep-mode configurations under a QoS constraint
   (>= 95% of UEs at >= 5 Mbps, with safety margin). Choices are validated on the simulator ("real network").
4. **Federated learning** - FedAvg traffic forecasting across cells vs local-only vs centralised training.

## Results (`python fiveg_twin.py`, seed 42; see `results.json`, `results_energy_qos.png`)
| Metric | Value |
|---|---|
| Twin fidelity (held-out) | R2 energy 0.98, R2 exposure 0.99, QoS MAE 0.015 |
| Energy saving vs all-cells-max-power | 24.8% overall (36.3% at 0-6h, 12.8% at 12-21h) |
| Exposure proxy reduction | 75.1% |
| QoS (UEs >= 5 Mbps) | 0.979 mean (baseline 0.995); target met in 97.5% of 120 scenarios |
| Traffic forecast MAE (UEs) | persistence 1.85, local 1.55, FedAvg 1.56, centralised 1.57 |

## Limitations (be explicit about them)
- Simplified 7-cell cluster, omnidirectional-equivalent antennas, no inter-cluster interference, synthetic traffic.
- The exposure value is a relative proxy, not a regulatory SAR/EMF compliance computation.
- In this synthetic setting FedAvg matches centralised training but is not better than local-only models;
  its benefit here is privacy (raw data stays in each cell) at no accuracy cost.
- Possible extensions: sectorised cells, 3GPP UMa/UMi channel models, ns-3 or Sionna, mission-critical
  slicing (URLLC), RL-based control inside the twin, closing the loop with the federated traffic forecast.

## Run
```
pip install numpy scikit-learn matplotlib
python fiveg_twin.py
```
