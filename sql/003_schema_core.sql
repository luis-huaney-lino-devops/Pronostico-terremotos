-- =====================================================================
-- 003 · Esquema CORE
--   observations   : cada observación NORMALIZADA (magnitud homogenizada a Mw,
--                    geometría PostGIS, país). 1 fila = 1 fuente ve el evento.
--   events         : evento CANÓNICO (físico). 1 fila = 1 terremoto real,
--                    con origen y magnitud preferidos tras deduplicar.
--   event_members  : enlaza observaciones -> evento canónico.
-- =====================================================================

CREATE TABLE IF NOT EXISTS core.observations (
    obs_id           BIGSERIAL PRIMARY KEY,
    source           TEXT        NOT NULL,
    source_event_id  TEXT        NOT NULL,
    origin_time      TIMESTAMPTZ NOT NULL,
    geom             geometry(Point, 4326) NOT NULL,   -- lon/lat WGS84
    depth_km         DOUBLE PRECISION,
    magnitude        DOUBLE PRECISION,                 -- magnitud original
    magnitude_type   TEXT,
    mw               DOUBLE PRECISION,                 -- magnitud homogenizada
    mw_method        TEXT,                             -- p.ej. 'direct', 'scordilis_ms'
    author           TEXT,
    place            TEXT,
    country          TEXT,
    quality          TEXT,
    canonical_id     BIGINT,                           -- FK lógica a core.events
    raw_id           BIGINT REFERENCES raw.source_events(id),
    UNIQUE (source, source_event_id)
);
CREATE INDEX IF NOT EXISTS ix_obs_time    ON core.observations (origin_time);
CREATE INDEX IF NOT EXISTS ix_obs_geom    ON core.observations USING GIST (geom);
CREATE INDEX IF NOT EXISTS ix_obs_canon   ON core.observations (canonical_id);
CREATE INDEX IF NOT EXISTS ix_obs_country ON core.observations (country);

-- Evento canónico (deduplicado): el terremoto físico único.
CREATE TABLE IF NOT EXISTS core.events (
    canonical_id       BIGSERIAL PRIMARY KEY,
    canonical_uuid     UUID NOT NULL DEFAULT uuid_generate_v4(),
    origin_time        TIMESTAMPTZ NOT NULL,           -- de la fuente preferida
    geom               geometry(Point, 4326) NOT NULL,
    depth_km           DOUBLE PRECISION,
    preferred_mw       DOUBLE PRECISION,
    preferred_mag      DOUBLE PRECISION,
    preferred_mag_type TEXT,
    preferred_source   TEXT,                           -- fuente elegida para el origen
    mag_source         TEXT,                           -- fuente elegida para la magnitud
    n_sources          INTEGER NOT NULL DEFAULT 1,
    country            TEXT,
    region             TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_evt_time    ON core.events (origin_time);
CREATE INDEX IF NOT EXISTS ix_evt_geom    ON core.events USING GIST (geom);
CREATE INDEX IF NOT EXISTS ix_evt_mw      ON core.events (preferred_mw);
CREATE INDEX IF NOT EXISTS ix_evt_country ON core.events (country);

-- Enlace observación -> evento canónico (N observaciones : 1 evento).
CREATE TABLE IF NOT EXISTS core.event_members (
    canonical_id BIGINT NOT NULL REFERENCES core.events(canonical_id) ON DELETE CASCADE,
    obs_id       BIGINT NOT NULL REFERENCES core.observations(obs_id) ON DELETE CASCADE,
    is_preferred BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (canonical_id, obs_id)
);
