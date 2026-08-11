-- =====================================================================
-- 004 · Esquema SPATIAL (geometrías de apoyo)
--   regions      : países / macro-regiones del Perú (bbox por ahora)
--   faults       : fallas activas (se poblará en Fase 3)
--   grid_cells   : malla 0.1° del Perú (se genera en Fase 3)
-- =====================================================================

CREATE TABLE IF NOT EXISTS spatial.regions (
    region_id  SERIAL PRIMARY KEY,
    name       TEXT NOT NULL,          -- 'Peru', 'Chile', 'Peru-Norte', ...
    kind       TEXT NOT NULL,          -- 'country' | 'macro' | 'custom'
    country    TEXT,
    geom       geometry(MultiPolygon, 4326) NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_regions_geom ON spatial.regions USING GIST (geom);

CREATE TABLE IF NOT EXISTS spatial.faults (
    fault_id   SERIAL PRIMARY KEY,
    name       TEXT,
    fault_type TEXT,                   -- 'reverse' | 'normal' | 'strike-slip' | 'subduction'
    source     TEXT,                   -- p.ej. 'INGEMMET Neotectónica'
    geom       geometry(MultiLineString, 4326)
);
CREATE INDEX IF NOT EXISTS ix_faults_geom ON spatial.faults USING GIST (geom);

CREATE TABLE IF NOT EXISTS spatial.grid_cells (
    grid_id   BIGSERIAL PRIMARY KEY,
    res_deg   DOUBLE PRECISION NOT NULL,   -- resolución (0.1)
    i_idx     INTEGER NOT NULL,
    j_idx     INTEGER NOT NULL,
    centroid  geometry(Point, 4326) NOT NULL,
    cell      geometry(Polygon, 4326) NOT NULL,
    UNIQUE (res_deg, i_idx, j_idx)
);
CREATE INDEX IF NOT EXISTS ix_grid_cell     ON spatial.grid_cells USING GIST (cell);
CREATE INDEX IF NOT EXISTS ix_grid_centroid ON spatial.grid_cells USING GIST (centroid);
