-- =====================================================================
-- 005 · Semilla de regiones (bounding boxes por país)
--   Aproximación inicial para etiquetar país por punto-en-polígono.
--   Se reemplazará por polígonos administrativos reales más adelante.
--   ST_MakeEnvelope(min_lon, min_lat, max_lon, max_lat, 4326)
-- =====================================================================
INSERT INTO spatial.regions (name, kind, country, geom)
SELECT v.name, 'country', v.name,
       ST_Multi(ST_MakeEnvelope(v.min_lon, v.min_lat, v.max_lon, v.max_lat, 4326))
FROM (VALUES
    ('Peru',     -81.5, -18.5, -68.0,   0.5),
    ('Ecuador',  -82.0,  -5.0, -75.0,   2.0),
    ('Chile',    -76.0, -56.0, -66.0, -17.0),
    ('Bolivia',  -70.0, -23.0, -57.5,  -9.0),
    ('Colombia', -82.0,  -4.5, -66.0,  13.0)
) AS v(name, min_lon, min_lat, max_lon, max_lat)
ON CONFLICT DO NOTHING;
