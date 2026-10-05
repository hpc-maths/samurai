// "/" focuses the search field of the header, as on most documentation sites.
document.addEventListener("keydown", (event) => {
  if (event.key !== "/" || event.ctrlKey || event.metaKey || event.altKey) return;
  const target = event.target;
  if (target.closest("input, textarea, select, [contenteditable]")) return;
  const field = document.querySelector(".sm-search input");
  if (!field || field.offsetParent === null) return;
  event.preventDefault();
  field.focus();
});

// On Read the Docs, the addons API lists the active versions: the version label
// of the header becomes a selector that keeps the reader on the same page.
document.addEventListener("readthedocs-addons-data-ready", (event) => {
  const data = event.detail.data();
  const current = data.versions.current;
  const active = data.versions.active;
  document.querySelectorAll("[data-sm-version]").forEach((cell) => {
    const select = document.createElement("select");
    select.setAttribute("aria-label", "Documentation version");
    for (const version of active) {
      const option = document.createElement("option");
      option.value = version.urls.documentation;
      option.textContent = version.slug;
      option.selected = version.slug === current.slug;
      select.append(option);
    }
    select.addEventListener("change", () => {
      const base = current.urls.documentation;
      const page = window.location.href.startsWith(base) ? window.location.href.slice(base.length) : "";
      window.location.href = select.value + page;
    });
    cell.replaceChildren(select);
  });
});
