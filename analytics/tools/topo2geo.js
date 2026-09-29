const topo = require('topojson-client');
const fs = require('fs');
const t = JSON.parse(fs.readFileSync('node_modules/world-atlas/land-50m.json'));
const g = topo.feature(t, t.objects.land);
fs.writeFileSync('land50.geojson', JSON.stringify(g));
console.log(g.type, g.features ? g.features.length : Object.keys(g));
