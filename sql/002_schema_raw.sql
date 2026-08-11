-- =====================================================================
-- 002 · Esquema RAW  (una fila = una observación cruda de UNA fuente)
--        Regla de oro: aquí NO se corrige ni se borra nada.
-- =====================================================================

-- Manifiesto de cada lote de ingesta (reproducibilidad).
CREATE TABLE IF NOT EXISTS raw.ingest_batches (
    batch_id      BIGSERIAL PRIMARY KEY,
    source        TEXT        NOT NULL,          -- 'usgs' | 'isc' | 'emsc' | 'iris' | 'igp'
    query_start   TIMESTAMPTZ,
    query_end     TIMESTAMPTZ,
    bbox          TEXT,                          -- bounding box consultado
    params        JSONB,                         -- parámetros exactos de la query
    n_events      INTEGER,
    fetched_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    tool_version  TEXT
);

-- Observaciones crudas, tal cual llegan de la fuente (campos nativos FDSN).
CREATE TABLE IF NOT EXISTS raw.source_events (
    id               BIGSERIAL PRIMARY KEY,
    source           TEXT        NOT NULL,       -- fuente
    source_event_id  TEXT        NOT NULL,       -- id nativo del evento en la fuente
    origin_time      TIMESTAMPTZ NOT NULL,
    latitude         DOUBLE PRECISION NOT NULL,
    longitude        DOUBLE PRECISION NOT NULL,
    depth_km         DOUBLE PRECISION,
    magnitude        DOUBLE PRECISION,
    magnitude_type   TEXT,
    author           TEXT,
    catalog          TEXT,
    contributor      TEXT,
    place            TEXT,
    raw_payload      JSONB,                      -- fila original completa
    batch_id         BIGINT REFERENCES raw.ingest_batches(batch_id),
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- misma fuente no debe duplicar el mismo id de evento:
    UNIQUE (source, source_event_id)
);

CREATE INDEX IF NOT EXISTS ix_raw_src_time  ON raw.source_events (origin_time);
CREATE INDEX IF NOT EXISTS ix_raw_src_src   ON raw.source_events (source);
CREATE INDEX IF NOT EXISTS ix_raw_src_mag   ON raw.source_events (magnitude);
