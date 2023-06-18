import paths
import plots
import reports

data_name = "glove-25"

# Plot lfd-vs-depth
trees_path = paths.REPORTS_DIR.joinpath(
    "mac_2",
    "trees",
)
output_base = trees_path.joinpath("plots")
output_base.mkdir(exist_ok=True)

for path in trees_path.iterdir():
    if "__" not in path.name:
        continue
    if data_name not in path.name:
        continue

    print(f"reading from {path.name} ...")
    tree: reports.TreeReport = reports.TreeReport.parse_file(path.joinpath("tree.json"))
    clusters = reports.load_tree(path)
    clusters_by_depth = [
        [c for c in clusters if len(c.name) == d + 1] for d in range(tree.max_depth)
    ]
    plots.plot_lfd_vs_depth(
        "violin",
        tree,
        clusters_by_depth,
        False,
        output_base,
    )

# Plot search timings
search_path = paths.REPORTS_DIR.joinpath(
    "mac_2",
)
output_base = search_path.joinpath("plots")
output_base.mkdir(exist_ok=True)

for path in search_path.iterdir():
    if not path.name.endswith("json"):
        continue
    if data_name not in path.name:
        continue

    r: reports.RnnReport = reports.RnnReport.parse_file(path)
    assert len(r.is_valid()) == 0
    r.plot(False, output_base)
