"""
Cross-event common and event-specific regions (Tables S6 and S10).

Per film and group: M[e, c] = mean over the group's participants of the event map
(median |IG| over event e's TRs, data column c). Common set = top 12 (5%) of the
equal-weight mean over events, without within-event normalisation. Event-specific
regions = an event's own top 12 minus the common set. The same definition is applied
to the TDC group. For comparison, a within-event z-scored version (z across the 246
columns, population SD) is also written. Network average = mean and sample SD of the
across-event ASD map over all regions of each Dev-Atlas system.

Data column c is labelled Brainnetome ID c (column 0 has no label).
"""
import numpy as np
import pandas as pd

from cr_common import DATA_IDS, K, OUT, SYSTEM_LONG, SYSTEM_ORDER, films_with, regions

R = regions()
rows = {k: [] for k in ("common", "spec", "spec_sum", "zcmp", "zdiff", "tdc", "tdc_set", "avg", "ev", "net")}


def info(c):
    if c not in R.index or c == 0:
        return dict(brainnetome_id=None, brainnetome_label="column 0 (no label)", anatomical_label="column 0 (no label)",
                    region="", system="", system_name="", devatlas_network_id=None, network_index="")
    r = R.loc[c]
    return dict(brainnetome_id=c, brainnetome_label=r.label, anatomical_label=r.anatomical_label, region=r.region,
                system=r.system, system_name=SYSTEM_LONG[r.system], devatlas_network_id=int(r.devatlas_id),
                network_index=r.subnetwork)


for film in films_with("event_maps_{film}.npz"):
    z = np.load(OUT / f"event_maps_{film}.npz")
    em, grp = z["maps"].astype(float), z["group"].astype(str)
    n_ev = em.shape[1]
    evs = [f"E{e + 1}" for e in range(n_ev)]
    res = {}
    for g in ("ASD", "TDC"):
        M = np.stack([em[grp == g, e, :].mean(axis=0) for e in range(n_ev)])          # E x 246
        Z = (M - M.mean(axis=1, keepdims=True)) / M.std(axis=1, keepdims=True)
        avgM, avgZ = M.mean(axis=0), Z.mean(axis=0)
        oM, oZ = np.argsort(avgM)[::-1], np.argsort(avgZ)[::-1]
        res[g] = dict(M=M, avgM=avgM, avgZ=avgZ, n=int((grp == g).sum()),
                      ev_top={ev: set(int(i) for i in np.argsort(M[e])[-K:]) for e, ev in enumerate(evs)},
                      rankM={int(c): i + 1 for i, c in enumerate(oM)}, rankZ={int(c): i + 1 for i, c in enumerate(oZ)},
                      raw=[int(c) for c in oM[:K]], zs=[int(c) for c in oZ[:K]])
    if 0 in res["ASD"]["raw"]:
        raise ValueError(f"{film}: unlabelled column 0 in the ASD common set")
    A, T = res["ASD"], res["TDC"]
    prim, pset, zset = A["raw"], set(A["raw"]), set(A["zs"])
    nin = lambda c: int(sum(c in A["ev_top"][ev] for ev in evs))  # noqa: E731
    lab = lambda c: info(c)["anatomical_label"]  # noqa: E731

    for c in prim:
        rec = dict(film=film, rank=A["rankM"][c], data_column=c, **info(c),
                   mean_ig_x1000=round(float(A["avgM"][c]) * 1e3, 4),
                   rank_in_z_average=A["rankZ"][c], mean_z=round(float(A["avgZ"][c]), 3))
        rec.update({f"top5_{ev}": c in A["ev_top"][ev] for ev in evs})
        rec.update(events_in_top5=", ".join(ev for ev in evs if c in A["ev_top"][ev]), n_events_in_top5=nin(c),
                   in_z_scored_set=c in zset)
        rows["common"].append(rec)
    for e, ev in enumerate(evs):
        order = [int(c) for c in np.argsort(A["M"][e])[::-1][:K]]
        spec = [c for c in order if c not in pset]
        for c in spec:
            rows["spec"].append(dict(film=film, event=ev, rank_in_event=order.index(c) + 1, **info(c),
                                     ig_in_event_x1000=round(float(A["M"][e, c]) * 1e3, 4),
                                     mean_ig_x1000=round(float(A["avgM"][c]) * 1e3, 4),
                                     events_in_top5=", ".join(x for x in evs if c in A["ev_top"][x])))
        rows["spec_sum"].append(dict(film=film, event=ev, n_common=K - len(spec), n_event_specific=len(spec),
                                     common_in_event=", ".join(lab(c) for c in order if c in pset),
                                     event_specific=", ".join(f"{lab(c)} [{info(c)['system']}]" for c in spec)))
    rows["zcmp"].append(dict(film=film, k=K, overlap_raw_vs_z=len(pset & zset),
                             jaccard=round(len(pset & zset) / len(pset | zset), 3),
                             raw_only=", ".join(lab(c) for c in prim if c not in zset),
                             z_only=", ".join(lab(c) for c in A["zs"] if c not in pset)))
    for c in sorted(pset ^ zset, key=lambda c: A["rankM"][c]):
        rows["zdiff"].append(dict(film=film, set="raw only" if c in pset else "z-scored only", **info(c),
                                  rank_in_raw_average=A["rankM"][c], rank_in_z_average=A["rankZ"][c],
                                  n_events_in_top5=nin(c)))
    tset = set(T["raw"])
    rows["tdc"].append(dict(film=film, n_ASD=A["n"], n_TDC=T["n"], overlap_with_ASD=len(tset & pset),
                            jaccard=round(len(tset & pset) / len(tset | pset), 3),
                            shared=", ".join(lab(c) for c in prim if c in tset),
                            ASD_only=", ".join(lab(c) for c in prim if c not in tset),
                            TDC_only=", ".join(lab(c) for c in T["raw"] if c not in pset)))
    for c in T["raw"]:
        rows["tdc_set"].append(dict(film=film, rank=T["rankM"][c], **info(c),
                                    mean_ig_x1000=round(float(T["avgM"][c]) * 1e3, 4), in_ASD_common_set=c in pset))
    for c in DATA_IDS:
        rows["avg"].append(dict(film=film, data_column=c, **info(c), circle_order=int(R.loc[c, "circle_order"]),
                                ASD_mean_ig=float(A["avgM"][c]), ASD_mean_z=float(A["avgZ"][c]),
                                TDC_mean_ig=float(T["avgM"][c]), TDC_mean_z=float(T["avgZ"][c]),
                                ASD_common=c in pset, ASD_common_z=c in zset, TDC_common=c in tset,
                                ASD_n_events_top5=nin(c)))
        for e, ev in enumerate(evs):
            for g in ("ASD", "TDC"):
                rows["ev"].append(dict(film=film, event=ev, group=g, n_participants=res[g]["n"], brainnetome_id=c,
                                       value=float(res[g]["M"][e, c]), in_top5_ASD=c in A["ev_top"][ev],
                                       in_top5_TDC=c in T["ev_top"][ev]))
    for s in SYSTEM_ORDER:
        mem = [c for c in DATA_IDS if R.loc[c, "system"] == s]
        v = A["avgM"][mem]
        rows["net"].append(dict(film=film, system=s, n_regions=len(mem), mean=float(v.mean()), sd=float(v.std(ddof=1)),
                                mean_x1000=round(float(v.mean()) * 1e3, 4), sd_x1000=round(float(v.std(ddof=1)) * 1e3, 4)))
    print(f"{film}: common regions " + ", ".join(R.loc[c, "label"] for c in prim))

names = {"common": "common_regions", "spec": "event_specific_regions", "spec_sum": "event_specific_summary",
         "zcmp": "z_scored_comparison", "zdiff": "z_scored_differences", "tdc": "tdc_common_overlap",
         "tdc_set": "tdc_common_regions", "avg": "average_map", "ev": "event_values", "net": "network_average"}
for k, n in names.items():
    if rows[k]:
        pd.DataFrame(rows[k]).to_csv(OUT / f"{n}.csv", index=False)
