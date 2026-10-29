-- Applied by PostgreSQL only when initializing a fresh data volume.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(512) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS episodic_memories (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    conv_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    full_text TEXT NOT NULL,
    embedding vector(512) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS episodic_memories_user_id_idx ON episodic_memories(user_id);

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT PRIMARY KEY,
    profile JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS reimbursement_cases (
    case_id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    conv_id TEXT NOT NULL,
    department TEXT NOT NULL DEFAULT 'engineering',
    state TEXT NOT NULL DEFAULT 'collecting_info',
    facts JSONB NOT NULL DEFAULT '{}'::jsonb,
    expenses JSONB NOT NULL DEFAULT '[]'::jsonb,
    validation JSONB NOT NULL DEFAULT '{}'::jsonb,
    budget_reserved BOOLEAN NOT NULL DEFAULT FALSE,
    claim_no TEXT UNIQUE,
    version INTEGER NOT NULL DEFAULT 0 CHECK (version >= 0),
    events JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS access_requests (
    request_id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    system_name TEXT NOT NULL,
    permission TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL DEFAULT 'draft',
    version INTEGER NOT NULL DEFAULT 0 CHECK (version >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS operations (
    operation_id TEXT PRIMARY KEY,
    actor_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    params JSONB NOT NULL,
    preview JSONB NOT NULL,
    snapshot_hash TEXT NOT NULL,
    resource_version INTEGER NOT NULL CHECK (resource_version >= 0),
    state TEXT NOT NULL DEFAULT 'pending',
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS operations_actor_state_idx ON operations(actor_id, state);

CREATE TABLE IF NOT EXISTS idempotency_records (
    actor_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    response JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT idempotency_records_actor_action_key_unique
        UNIQUE (actor_id, action_type, idempotency_key)
);

CREATE TABLE IF NOT EXISTS operation_audit_events (
    id BIGSERIAL PRIMARY KEY,
    operation_id TEXT,
    event_type TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    detail JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- Nullable operation_id allows recording proposals and rejected prechecks
-- before an operation has been created.
CREATE INDEX IF NOT EXISTS operation_audit_events_operation_id_idx
    ON operation_audit_events(operation_id);
