/**
 * Homepage controller — the Phase 2 surface.
 *
 * Loads three things independently: the live stats readout, "Picked for you" for
 * this browser's anonymous id, and the catalogue grid. Independently on purpose —
 * if the recommendation call fails, the catalogue should still render. A shop that
 * shows nothing because its recommender is down is worse than one that shows
 * products without personalisation.
 */
(function (window, document) {
  "use strict";

  var api = window.TesschApi;
  var render = window.TesschRender;
  var tracker = window.TesschTracker;

  var visitorId = tracker.visitorId();
  document.getElementById("visitor-id").textContent = visitorId;

  document
    .getElementById("reset-visitor")
    .addEventListener("click", function () {
      // The demo needs a way to become a different person on demand. Without it,
      // showing that two visitors get different lists means clearing site data by
      // hand between takes.
      tracker.resetVisitor();
      window.location.reload();
    });

  // ── Stats readout ─────────────────────────────────────────────────────────
  function loadStats() {
    Promise.all([
      api.health().catch(function () {
        return null;
      }),
      api.provenance().catch(function () {
        return null;
      }),
      api.model().catch(function () {
        return null;
      }),
    ]).then(function (results) {
      var health = results[0];
      var provenance = results[1];
      var model = results[2];

      if (health) {
        document.getElementById("stat-products").textContent = health.products;
        document.getElementById("stat-events").textContent =
          health.view_events.toLocaleString("en-IN");
      }
      if (provenance) {
        // Stated as a share, not a raw count, because the honest headline is that
        // most of this data is simulated — and rounding it away would be a way of
        // not saying so.
        var pct = (provenance.live_share * 100).toFixed(1);
        document.getElementById("stat-live").textContent =
          provenance.live.views + " views (" + pct + "%)";
      }
      if (model && model.trained) {
        document.getElementById("stat-coverage").textContent =
          model.products_with_neighbours + " / " + model.n_products;
      } else {
        document.getElementById("stat-coverage").textContent = "not trained yet";
      }
    });
  }

  // ── Picked for you ────────────────────────────────────────────────────────
  function loadPicked() {
    var grid = document.getElementById("picked");
    var why = document.getElementById("picked-why");
    render.showSkeletons(grid, 4);

    api
      .forUser(visitorId, 8)
      .then(function (response) {
        render.strategyBadge(document.getElementById("picked-strategy"), response);
        why.textContent = response.reason || "";

        if (!response.items.length) {
          render.showNote(
            grid,
            "Nothing recorded yet. Open a product below and stay on it for a few seconds — this section fills in as soon as there is something to rank.",
            null
          );
          return;
        }

        render.fillGrid(grid, response.items, {
          showScore: false,
          note: function (product) {
            // The dwell time is the whole point of this section, so it is shown
            // on the card rather than hidden behind a tooltip.
            if (!product.total_seconds) return null;
            return (
              Math.round(product.total_seconds) +
              "s" +
              (product.view_count > 1 ? " · " + product.view_count + " visits" : "")
            );
          },
        });
      })
      .catch(function (error) {
        render.showNote(
          grid,
          "Could not load your picks: " + error.message,
          "warn"
        );
      });
  }

  // ── Catalogue ─────────────────────────────────────────────────────────────
  function loadCatalogue() {
    var grid = document.getElementById("catalogue");
    render.showSkeletons(grid, 8);

    api
      .products({ limit: 100 })
      .then(function (response) {
        document.getElementById("catalogue-count").textContent =
          response.count + " of " + response.total;
        render.fillGrid(grid, response.items);
      })
      .catch(function (error) {
        render.showNote(
          grid,
          "Could not load the catalogue: " +
            error.message +
            " — check that the server is running and the database has been created.",
          "warn"
        );
      });
  }

  loadStats();
  loadPicked();
  loadCatalogue();
})(window, document);
