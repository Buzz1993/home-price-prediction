// Exposes Leaflet on `window` so the UMD `leaflet.markercluster` plugin — which
// references the *global* `L` rather than importing it as a module — can attach
// its `markerClusterGroup` to the same Leaflet instance. This module is imported
// before the plugin (see property-map-view.tsx) so the global is set first;
// without it the plugin evaluates with no global Leaflet and throws
// "ReferenceError: L is not defined" in a bundled app (there is no script-tag L).
//
// Client-only: property-map-view.tsx is loaded via next/dynamic (ssr: false), and
// the window guard keeps this a no-op if it is ever evaluated on the server.
import L from "leaflet";

if (typeof window !== "undefined") {
  (window as unknown as { L: typeof L }).L = L;
}

export {};
