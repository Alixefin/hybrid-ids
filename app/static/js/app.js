// Small progressive enhancements. There are no inline scripts: the Content-Security-Policy
// only allows 'self' and cdn.jsdelivr.net, so chart data is passed in
// <script type="application/json"> blocks.
(function () {
  "use strict";

  // Confirmation prompts for destructive buttons.
  document.addEventListener("click", function (e) {
    const el = e.target.closest("[data-confirm]");
    if (el && !window.confirm(el.getAttribute("data-confirm"))) {
      e.preventDefault();
      e.stopPropagation();
    }
  }, true);

  // "Scoring…" overlay for long synchronous detection runs.
  document.querySelectorAll("form[data-busy]").forEach(function (form) {
    form.addEventListener("submit", function () {
      const div = document.createElement("div");
      div.className = "busy-overlay";
      div.innerHTML = '<div class="text-center"><div class="spinner-border mb-2" role="status"></div><div></div></div>';
      div.querySelector("div > div:last-child").textContent = form.getAttribute("data-busy");
      document.body.appendChild(div);
    });
  });

  // Select-all checkboxes.
  document.querySelectorAll("[data-select-all]").forEach(function (box) {
    box.addEventListener("change", function () {
      const name = box.getAttribute("data-select-all");
      document.querySelectorAll('input[name="' + name + '"]').forEach(function (c) { c.checked = box.checked; });
    });
  });

  if (typeof Chart === "undefined") return;

  const palette = ["#dc3545", "#fd7e14", "#ffc107", "#0d6efd", "#6f42c1", "#20c997", "#6c757d", "#198754", "#0dcaf0"];
  function readJson(id) {
    const el = document.getElementById(id);
    return el ? JSON.parse(el.textContent) : null;
  }
  function chart(id, type, series, opts) {
    const el = document.getElementById(id);
    if (!el || !series || !series.values || !series.values.length) return;
    new Chart(el, {
      type: type,
      data: { labels: series.labels, datasets: [{ data: series.values, backgroundColor: opts && opts.colors || palette,
        borderColor: type === "line" ? "#0d6efd" : undefined, fill: false, tension: 0 }] },
      options: Object.assign({ responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: type === "doughnut", position: "bottom", labels: { boxWidth: 12 } } } },
        opts && opts.options || {})
    });
  }

  const dash = readJson("dashboard-data");
  if (dash) {
    chart("chartByClass", "doughnut", dash.by_class);
    chart("chartRoutes", "doughnut", dash.routes, { colors: ["#198754", "#6c757d", "#dc3545", "#ffc107"] });
    chart("chartByAction", "bar", dash.by_action, { options: { indexAxis: "y" } });
    chart("chartPerDay", "line", dash.per_day, { options: { scales: { y: { beginAtZero: true } } } });
  }
  const result = readJson("result-data");
  if (result) chart("chartResultClasses", "doughnut", result);
})();
