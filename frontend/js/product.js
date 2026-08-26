/**
 * Product page controller — starts the dwell timer, renders the product, and loads
 * the Phase 3 "You might also like" list.
 *
 * The timer starts before the recommendations are fetched. That ordering matters:
 * if it started after, every view would silently lose however long the network
 * took, and slow connections would under-report attention exactly where the data
 * is most valuable.
 */
(function (window, document) {
  "use strict";

  var api = window.TesschApi;
  var render = window.TesschRender;
  var trackerLib = window.TesschTracker;

  var productId = decodeURIComponent(
    window.location.pathname.replace(/^\/product\//, "").replace(/\/$/, "")
  ).toUpperCase();

  document.getElementById("visitor-id").textContent = trackerLib.visitorId();

  // ── Start measuring immediately ───────────────────────────────────────────
  var tracker = trackerLib.create();
  // `surface` records how the visitor arrived. It is not used by either
  // recommender yet, but position and surface are what a later analysis of
  // exposure bias would need, and they cannot be reconstructed after the fact.
  var cameFromRecommendation = /[?&]via=rec/.test(window.location.search);
  tracker.start(productId, {
    surface: cameFromRecommendation ? "recommendation" : "product_page",
  });
  render.attentionMeter(tracker);

  // ── Product detail ────────────────────────────────────────────────────────
  function specRow(label, value) {
    if (!value) return null;
    var row = document.createElement("tr");
    var th = document.createElement("th");
    th.scope = "row";
    th.textContent = label;
    var td = document.createElement("td");
    td.textContent = value;
    row.appendChild(th);
    row.appendChild(td);
    return row;
  }

  api
    .product(productId)
    .then(function (product) {
      document.title = product.product_name + " — TESSCH";
      document.getElementById("detail-code").textContent =
        product.product_id + " / " + (product.category || "").toUpperCase();
      document.getElementById("detail-name").textContent = product.product_name;
      document.getElementById("detail-price").textContent = render.rupees(
        product.price
      );
      document.getElementById("detail-desc").textContent =
        product.description || "";

      var tileHost = document.getElementById("detail-tile");
      tileHost.textContent = "";
      tileHost.appendChild(render.tile(product, { showCode: false }));

      var spec = document.getElementById("detail-spec");
      [
        ["Category", product.category],
        ["Upper", product.upper_type],
        ["Colour", product.color],
        ["Style", product.style],
        ["Material", product.material],
        ["Collection", product.collection],
        [
          "Availability",
          product.availability === false ? "Out of stock" : "In stock",
        ],
      ].forEach(function (pair) {
        var row = specRow(pair[0], pair[1]);
        if (row) spec.appendChild(row);
      });
    })
    .catch(function (error) {
      document.getElementById("detail-name").textContent = "Product not found";
      document.getElementById("detail-desc").textContent =
        error.message + " Try returning to the catalogue.";
    });

  // ── You might also like ───────────────────────────────────────────────────
  var grid = document.getElementById("similar");
  render.showSkeletons(grid, 4);

  api
    .similarTo(productId, 6)
    .then(function (response) {
      render.strategyBadge(document.getElementById("similar-strategy"), response);
      document.getElementById("similar-why").textContent = response.reason || "";

      if (!response.items.length) {
        render.showNote(
          grid,
          "No similar products yet. This one needs to be viewed alongside others before the model can relate it to anything.",
          null
        );
        return;
      }

      render.fillGrid(grid, response.items, { showScore: true });

      // Tag outbound links so the next page records that it was reached from a
      // recommendation. This is how a click-through rate becomes computable later
      // without changing the schema.
      Array.prototype.forEach.call(grid.querySelectorAll("a.card"), function (a) {
        a.href += "?via=rec";
      });
    })
    .catch(function (error) {
      render.showNote(
        grid,
        "Could not load similar products: " + error.message,
        "warn"
      );
    });
})(window, document);
