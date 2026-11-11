# Cross-event common regions

Run in order from this folder. Inputs: the fold outputs of the training scripts
(`saved_models/fold*/ig_roi_milout.npy`, `labels.npy`, `subject_ids.npy`) and the
attention table (`figures/attn_significance.csv`) of each film. Paths are set in
`cr_common.py` and can be overridden with environment variables listed there.
Outputs go to `<RUN_DIR>/common_regions/`.

| Script | Output | Manuscript |
|---|---|---|
| `cr1_event_maps.py` | `event_maps_<film>.npz`: per-participant median \|IG\| per event, keyed by subject ID | input to all below |
| `cr2_common_regions.py` | `common_regions.csv`, `event_specific_regions.csv`, `event_specific_summary.csv`, `network_average.csv`, `tdc_common_regions.csv`, `tdc_common_overlap.csv`, `average_map.csv`, `event_values.csv`, `z_scored_comparison.csv` | Tables S6, S10 |
| `cr3_surfaces.py` | `surface_network_common_<film>.jpg` | Figures 3C, 5B (middle) |
| | `surface_common_vs_event_<film>.jpg`, `surface_common_<film>.jpg` | Figures 3D, 5C |
| `cr4_radars.py` | `radar_common_<film>.jpg`, `radar_common_events_<film>.jpg` | Figures 3C, 5B (right) |
| | `radar_events_<film>.jpg` | Figures S5, S10 |
| `cr5_schematic.py` | `schematic_<film>.jpg` | Figures 3C, 5B (left) |

Definitions: for each event, the ASD group map is the mean over ASD participants of
each participant's median \|IG\| within the event window. Common regions are the top 5%
(12 of 246) of the equal-weight mean of these maps across events, without within-event
normalisation; event-specific regions are an event's own top 5% minus the common set.
Network average: mean and SD of the across-event map over all regions of each Dev-Atlas
system. Data column c is labelled Brainnetome ID c.

Region labels and Dev-Atlas assignments: `assets/mapping/brainnetome_devatlas.csv`.
`cr3_surfaces.py` also needs the Brainnetome 246 volume (`BN_Atlas_246_1mm.nii`, from
https://atlas.brainnetome.org), placed in `assets/atlas/` or given with `BN_ATLAS`.
