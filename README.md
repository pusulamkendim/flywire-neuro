# FlyWire Neuro — Neurotransmitter Systems Analysis & Signal Propagation Simulations

Comprehensive neurotransmitter analysis and neural signal propagation simulations using the complete adult *Drosophila melanogaster* brain connectome (FlyWire v783).

## Key Findings

- **Six neurotransmitter systems** mapped across 139,255 neurons and 50M+ synapses with region-specific dominance patterns (dopamine→Mushroom Body, serotonin→Antennal Lobe, octopamine→Medulla/Lobula, GABA→Ellipsoid Body)
- **Signal propagation simulations** reveal modality-specific processing: olfactory signals reach motor output in 7 steps, gustatory in 1 step; taste bypasses memory circuits entirely
- **Punishment-before-reward priority**: PPL1 (punishment, 16 neurons) activates 2-3 steps before PAM (reward, 307 neurons) — confirmed across 400 parameter combinations with **0% reversal rate**
- **Hebbian learning simulation** on real connectome weights demonstrates odor-reward/punishment association, extinction dynamics, and sparse coding specificity

## Novel Contribution

**PPL1 temporal priority** over PAM is a structural property of the connectome, not a model artifact:
- PPL1 receives 11.6× more input per neuron (4,804 vs 414 synapses/neuron)
- PPL1 receives more direct ACh excitation (29% vs 16%); PAM depends on other dopamine neurons (70%)
- Robust across all tested decay (0.0–0.8), gain (0.5–5.0), threshold (0.01–0.5), and network filtering parameters
- Consistent across 10 different random odor representations
- Persists even when all GABA inhibition is removed

This finding has not been directly reported in the literature and is consistent with an evolutionary "threat-first" processing architecture.

## Data Sources

### FlyWire Connectome v783 ([Zenodo](https://zenodo.org/records/10676866))

| File | Size | Description |
|------|------|-------------|
| `proofread_connections_783.feather` | 813 MB | 16.8M neuron-neuron connections with synapse counts and NT probabilities |
| `neuron_annotations.tsv` | 31 MB | 139,244 neuron annotations (type, class, NT, lineage) |

Neuron annotations from [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations) (Schlegel et al., Nature 2024).

NT predictions by the [Synister CNN model](https://github.com/funkelab/synister) — 87% per-synapse, 94% per-neuron accuracy (Eckstein & Bates et al., Cell 2024).

## Analysis Pipeline

### NT Characterization (Analyses 02–11)

| Script | Analysis | Key Finding |
|--------|----------|-------------|
| `02_nt_distribution.py` | Synapse-weighted NT distribution across brain regions | ACh 48.4%, GABA 23.1%, Glut 19.2%, DA 4.4%, OCT 2.7%, SER 2.1% |
| `03_dopamine_serotonin.py` | DA vs SER neuron profiles and target regions | DA: 96.5% central, targets MB; SER: 68% sensory, targets AL |
| `05_da_ser_interaction.py` | DA↔SER cross-talk and bridge neurons | Systems avoid each other (0.21× expected); 19,235 bridge neurons |
| `06_reward_vs_punishment.py` | PAM (reward) vs PPL1 (punishment) circuit | 91% of Kenyon Cells receive from both; PPL1 10× stronger per neuron |
| `07_serotonin_role.py` | Serotonin's role in olfactory processing | 481× more synapses in AL than reward/punishment circuit |
| `08_octopamine.py` | Octopamine — visual system modulator | 48% photoreceptors; 58% synapses in visual areas; zero GABA interaction |
| `09_gaba_inhibition.py` | GABA inhibition patterns | EB 77.6% GABA dominant; 19.7% self-inhibition (disinhibition) |
| `10_acetylcholine.py` | Acetylcholine — primary excitatory NT | 86K neurons (62%); MB 90% ACh dominant |
| `11_glutamate.py` | Glutamate — dual excitatory/inhibitory role | FB 44.7%, PB 48.5%; Glut-MBONs provide reward feedback |

### Signal Propagation Simulations (Analyses 12–13)

| Script | Simulation | Key Finding |
|--------|------------|-------------|
| `12_signal_propagation.py` | Olfactory signal propagation (684 ORNs → brain) | PPL1 at t+4, PAM at t+7; E/I balance stabilizes at 5.5:1 |
| `13_taste_propagation.py` | Gustatory signal propagation (408 neurons → brain) | Motor at t+1 (vs t+7 for odor); Kenyon Cells never activated |

### Learning & Priority Analysis (Analyses 14–16)

| Script | Analysis | Key Finding |
|--------|----------|-------------|
| `14_learning_simulation.py` | Hebbian reward/punishment learning on real KC→MBON weights | 10-trial learning; extinction 81% in 15 trials; control odor unaffected |
| `15_ppl1_priority.py` | PPL1 temporal priority — 7 independent tests | 10/10 odors PPL1 first; structural not inhibition-based |
| `16_sensitivity_analysis.py` | Parameter sensitivity — 400 combinations | PPL1 first: 70.5%, PAM first: 0.0%, Tie: 29.5% |

## Setup

```bash
git clone https://github.com/pusulamkendim/flywire-neuro.git
cd flywire-neuro
python3 -m venv venv
source venv/bin/activate
pip install pandas pyarrow matplotlib seaborn
```

Download data files from [Zenodo](https://zenodo.org/records/10676866) into `data/`.
Download neuron annotations from [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations) into `data/`.

## Project Structure

```
flywire-neuro/
├── data/                                    (not tracked — download separately)
│   ├── proofread_connections_783.feather
│   ├── neuron_annotations.tsv
│   └── hemibrain_meta.csv
├── results/
│   ├── *.png                                (visualization outputs)
│   └── *.md                                 (detailed analysis reports in Turkish)
├── 02_nt_distribution.py                    (NT distribution across brain regions)
├── 03_dopamine_serotonin.py                 (DA vs SER profiling)
├── 04_visualize.py                          (6 charts: pie, bar, heatmap)
├── 05_da_ser_interaction.py                 (DA↔SER interaction + bridge neurons)
├── 06_reward_vs_punishment.py               (PAM reward vs PPL1 punishment)
├── 07_serotonin_role.py                     (Serotonin in olfactory processing)
├── 08_octopamine.py                         (Octopamine — visual modulator)
├── 09_gaba_inhibition.py                    (GABA inhibition analysis)
├── 10_acetylcholine.py                      (Acetylcholine — primary excitatory)
├── 11_glutamate.py                          (Glutamate — dual role)
├── 12_signal_propagation.py                 (Olfactory signal propagation simulation)
├── 13_taste_propagation.py                  (Gustatory signal propagation simulation)
├── 14_learning_simulation.py                (Reward/punishment learning simulation)
├── 15_ppl1_priority.py                      (PPL1 temporal priority analysis)
├── 16_sensitivity_analysis.py               (Parameter sensitivity analysis)
└── README.md
```

## References

- Dorkenwald et al. (2024) "Neuronal wiring diagram of an adult brain" — *Nature*
- Schlegel et al. (2024) "Whole-brain annotation and multi-connectome cell typing" — *Nature*
- Lin, Yang et al. (2024) "Network Statistics of the Whole-Brain Connectome" — *Nature*
- Eckstein, Bates et al. (2024) "Neurotransmitter classification from EM images" — *Cell*
- Li et al. (2020) "The connectome of the adult Drosophila mushroom body" — *eLife*
- Aso et al. (2014) "The neuronal architecture of the mushroom body" — *eLife*
- Hulse et al. (2021) "A connectome of the Drosophila central complex" — *eLife*
- Winding et al. (2023) "The connectome of an insect brain" — *Science*

## License

MIT
