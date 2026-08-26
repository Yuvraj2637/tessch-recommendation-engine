/**
 * Thin client for the backend API.
 *
 * Every call goes through `request`, which does one thing worth doing carefully:
 * it turns a failed response into an Error carrying the backend's own `detail`
 * message. The API is deliberate about explaining rejections — "duration_seconds:
 * below the 1.0s floor" rather than "Bad Request" — and discarding that in the
 * client would waste the effort.
 */
(function (window) {
  "use strict";

  function request(path, options) {
    var opts = options || {};
    return window
      .fetch(path, {
        method: opts.method || "GET",
        headers: opts.body ? { "Content-Type": "application/json" } : undefined,
        body: opts.body ? JSON.stringify(opts.body) : undefined,
      })
      .then(function (response) {
        return response
          .json()
          .catch(function () {
            return {};
          })
          .then(function (payload) {
            if (!response.ok) {
              var detail =
                payload && payload.detail
                  ? payload.detail
                  : response.status + " " + response.statusText;
              var error = new Error(detail);
              error.status = response.status;
              throw error;
            }
            return payload;
          });
      });
  }

  function query(params) {
    var parts = [];
    Object.keys(params || {}).forEach(function (key) {
      var value = params[key];
      if (value !== null && value !== undefined && value !== "") {
        parts.push(encodeURIComponent(key) + "=" + encodeURIComponent(value));
      }
    });
    return parts.length ? "?" + parts.join("&") : "";
  }

  window.TesschApi = {
    health: function () {
      return request("/api/health");
    },
    provenance: function () {
      return request("/api/provenance");
    },
    model: function () {
      return request("/api/recommendations/model");
    },
    products: function (params) {
      return request("/api/products" + query(params));
    },
    product: function (productId) {
      return request("/api/products/" + encodeURIComponent(productId));
    },
    forUser: function (userId, topK) {
      return request(
        "/api/recommendations/user/" +
          encodeURIComponent(userId) +
          query({ top_k: topK })
      );
    },
    similarTo: function (productId, topK) {
      return request(
        "/api/recommendations/similar/" +
          encodeURIComponent(productId) +
          query({ top_k: topK })
      );
    },
    popular: function (topK) {
      return request("/api/recommendations/popular" + query({ top_k: topK }));
    },
    rebuild: function () {
      return request("/api/recommendations/rebuild", { method: "POST" });
    },
  };
})(window);
