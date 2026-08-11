-- =====================================================================
-- 001 · Extensiones y esquemas base
-- =====================================================================
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- raw     : datos crudos por fuente, NUNCA se modifican (append-only)
-- core    : catálogo normalizado + eventos canónicos (deduplicados)
-- spatial : geometrías de apoyo (regiones, fallas, grid)
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS spatial;

COMMENT ON SCHEMA raw     IS 'Datos crudos por fuente (IGP/USGS/ISC/EMSC/IRIS). Inmutable.';
COMMENT ON SCHEMA core    IS 'Catálogo normalizado y eventos canónicos deduplicados.';
COMMENT ON SCHEMA spatial IS 'Regiones, fallas, placas, grid espacial.';
