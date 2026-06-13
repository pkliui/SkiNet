// Make nbsphinx code-input cells collapsible (collapsed by default).
// For each `.nbinput` container, the inner code area is wrapped in a
// `.nb-collapse-body` and a "Show code" / "Hide code" toggle button is
// inserted before it. Paired with collapse_nbinput.css.
(function () {
  function setup() {
    var inputs = document.querySelectorAll(".nbinput");
    inputs.forEach(function (cell) {
      if (cell.dataset.nbCollapse === "done") return;
      cell.dataset.nbCollapse = "done";

      // The actual code lives in the container after the prompt column.
      var body = cell.querySelector(".input_area") || cell.querySelector(".highlight");
      if (!body) return;

      // Wrap the body so we can toggle just the code, not the prompt.
      var wrapper = document.createElement("div");
      wrapper.className = "nb-collapse-body";
      body.parentNode.insertBefore(wrapper, body);
      wrapper.appendChild(body);

      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "nb-collapse-toggle";
      btn.setAttribute("aria-expanded", "false");
      btn.textContent = "Show code";
      btn.addEventListener("click", function () {
        var expanded = wrapper.classList.toggle("nb-expanded");
        btn.setAttribute("aria-expanded", expanded ? "true" : "false");
        btn.textContent = expanded ? "Hide code" : "Show code";
      });
      cell.insertBefore(btn, wrapper);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setup);
  } else {
    setup();
  }
})();
