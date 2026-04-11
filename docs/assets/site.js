async function loadData() {
  const response = await fetch("./data/status.json", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`failed to load status.json: ${response.status}`);
  }
  return response.json();
}

function chip(text, cls) {
  return `<span class="chip ${cls || ""}">${text}</span>`;
}

function yesNo(value) {
  return `<span class="${value ? "yes" : "no"}">${value ? "yes" : "no"}</span>`;
}

function uniqueValues(rows, key) {
  return [...new Set(rows.map((row) => row[key]).filter(Boolean))].sort();
}

function fillSelect(select, values) {
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  }
}

function renderSummary(summary) {
  const cards = [
    ["Benchmarks", summary.benchmark_count],
    ["Deliverable Now", summary.deliverable_now_count],
    ["Review Required", summary.review_required_count],
    ["Parity Covered", summary.parity_covered_count],
    ["New Kernel", summary.implementation_strategy_counts.new_kernel || 0],
    ["Reuse Existing", summary.implementation_strategy_counts.reuse_existing || 0],
    ["Specialize Existing", summary.implementation_strategy_counts.specialize_existing || 0],
    ["Blocked Review", summary.implementation_strategy_counts.blocked_review || 0],
  ];
  document.getElementById("summary-grid").innerHTML = cards
    .map(
      ([label, value]) =>
        `<article class="summary-card"><span class="summary-label">${label}</span><strong class="summary-value">${value}</strong></article>`,
    )
    .join("");
}

function renderKernelTable(rows) {
  document.getElementById("kernel-table").innerHTML = rows
    .map((row) => {
      const statuses = Object.entries(row.status_counts)
        .map(([key, value]) => chip(`${key}: ${value}`, key))
        .join("");
      const strategies = Object.entries(row.strategy_counts)
        .map(([key, value]) => chip(`${key}: ${value}`, key))
        .join("");
      const benchmarks = row.benchmark_ids.map((id) => chip(id)).join("");
      const opKinds = row.op_kinds.map((value) => chip(value)).join("");
      return `
        <tr>
          <td><strong>${row.kernel}</strong></td>
          <td><span class="muted">${row.source_path || "-"}</span></td>
          <td>${benchmarks}</td>
          <td>${statuses}</td>
          <td>${strategies}</td>
          <td>${opKinds}</td>
          <td>${yesNo(row.parity_covered)}</td>
        </tr>
      `;
    })
    .join("");
}

function renderBacklogTable(rows) {
  document.getElementById("backlog-table").innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td><strong>${row.id}</strong></td>
          <td>${row.name}</td>
          <td>${chip(row.op_kind)}</td>
          <td>${chip(row.status, row.status)}</td>
          <td>${chip(row.implementation_strategy, row.implementation_strategy)}</td>
          <td>${yesNo(row.deliverable_now)}</td>
          <td>${yesNo(row.review_required)}</td>
        </tr>
      `,
    )
    .join("");
}

function renderBenchmarkTable(rows) {
  document.getElementById("benchmark-table").innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td><strong>${row.id}</strong></td>
          <td>${row.name}</td>
          <td>${chip(row.op_kind)}</td>
          <td>${chip(row.status, row.status)}</td>
          <td>${row.candidate_kernel ? `<strong>${row.candidate_kernel}</strong>` : `<span class="muted">unmapped</span>`}</td>
          <td>${chip(row.implementation_strategy, row.implementation_strategy)}</td>
          <td>${yesNo(row.deliverable_now)}</td>
          <td>${yesNo(row.review_required)}</td>
        </tr>
      `,
    )
    .join("");
}

function applyFilters(data) {
  const kernelSearch = document.getElementById("kernel-search").value.trim().toLowerCase();
  const kernelStatus = document.getElementById("kernel-status-filter").value;
  const kernelStrategy = document.getElementById("kernel-strategy-filter").value;
  const kernelParity = document.getElementById("kernel-parity-filter").value;

  const kernels = data.kernels.filter((row) => {
    const matchesSearch =
      !kernelSearch ||
      row.kernel.toLowerCase().includes(kernelSearch) ||
      row.benchmark_ids.some((id) => id.toLowerCase().includes(kernelSearch));
    const matchesStatus = !kernelStatus || row.status_counts[kernelStatus];
    const matchesStrategy = !kernelStrategy || row.strategy_counts[kernelStrategy];
    const matchesParity =
      !kernelParity || (kernelParity === "yes" ? row.parity_covered : !row.parity_covered);
    return matchesSearch && matchesStatus && matchesStrategy && matchesParity;
  });
  renderKernelTable(kernels);

  const backlogStatus = document.getElementById("backlog-status-filter").value;
  const backlogStrategy = document.getElementById("backlog-strategy-filter").value;
  const backlog = data.backlog.filter(
    (row) =>
      (!backlogStatus || row.status === backlogStatus) &&
      (!backlogStrategy || row.implementation_strategy === backlogStrategy),
  );
  renderBacklogTable(backlog);

  const benchmarkSearch = document.getElementById("benchmark-search").value.trim().toLowerCase();
  const benchmarkStatus = document.getElementById("benchmark-status-filter").value;
  const benchmarkStrategy = document.getElementById("benchmark-strategy-filter").value;
  const benchmarkOp = document.getElementById("benchmark-op-filter").value;
  const benchmarks = data.benchmarks.filter((row) => {
    const searchText = [
      row.id,
      row.name,
      row.op_kind,
      row.candidate_kernel || "",
      ...(row.local_kernels || []),
    ]
      .join(" ")
      .toLowerCase();
    return (
      (!benchmarkSearch || searchText.includes(benchmarkSearch)) &&
      (!benchmarkStatus || row.status === benchmarkStatus) &&
      (!benchmarkStrategy || row.implementation_strategy === benchmarkStrategy) &&
      (!benchmarkOp || row.op_kind === benchmarkOp)
    );
  });
  renderBenchmarkTable(benchmarks);
}

function bindFilters(data) {
  const statusValues = uniqueValues(data.benchmarks, "status");
  const strategyValues = uniqueValues(data.benchmarks, "implementation_strategy");
  const opValues = uniqueValues(data.benchmarks, "op_kind");

  fillSelect(document.getElementById("kernel-status-filter"), statusValues);
  fillSelect(document.getElementById("kernel-strategy-filter"), strategyValues);
  fillSelect(document.getElementById("backlog-status-filter"), statusValues);
  fillSelect(document.getElementById("backlog-strategy-filter"), strategyValues);
  fillSelect(document.getElementById("benchmark-status-filter"), statusValues);
  fillSelect(document.getElementById("benchmark-strategy-filter"), strategyValues);
  fillSelect(document.getElementById("benchmark-op-filter"), opValues);

  for (const id of [
    "kernel-search",
    "kernel-status-filter",
    "kernel-strategy-filter",
    "kernel-parity-filter",
    "backlog-status-filter",
    "backlog-strategy-filter",
    "benchmark-search",
    "benchmark-status-filter",
    "benchmark-strategy-filter",
    "benchmark-op-filter",
  ]) {
    document.getElementById(id).addEventListener("input", () => applyFilters(data));
    document.getElementById(id).addEventListener("change", () => applyFilters(data));
  }
}

loadData()
  .then((data) => {
    document.getElementById("generated-at").textContent = data.generated_at_utc;
    document.getElementById("workbook-path").textContent = data.source.workbook_path;
    document.getElementById("pto-root").textContent = data.source.pto_kernels_root;
    renderSummary(data.summary);
    bindFilters(data);
    applyFilters(data);
  })
  .catch((error) => {
    document.body.innerHTML = `<main class="page"><section class="panel"><h1>Failed to load panel</h1><p>${error.message}</p></section></main>`;
  });
