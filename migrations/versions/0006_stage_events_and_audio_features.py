"""stage_events and audio_features

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-14

* stage_events: one row per timed pipeline step of a run attempt (queue wait, cache, VAD, per-chunk features,
  encoder, decoder, scoring, database). Read by the dashboard.
* audio_features: per-file descriptive features (level, speech ratio, SNR estimate, spectral centroid, chunk and
  padding counts), computed once per audio file.
"""
from alembic import op
import sqlalchemy as sa


revision = '0006'
down_revision = '0005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'stage_events',
        sa.Column('id', sa.BigInteger().with_variant(sa.Integer(), 'sqlite'), primary_key=True, autoincrement=True),
        sa.Column('run_id', sa.String(length=32),
                  sa.ForeignKey('transcription_runs.id', name='fk_stage_events_run_id', ondelete='CASCADE'),
                  nullable=False),
        sa.Column('attempt', sa.Integer(), nullable=False),
        sa.Column('seq', sa.Integer(), nullable=False),
        sa.Column('parent_seq', sa.Integer(), nullable=True),
        sa.Column('stage', sa.String(length=64), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('duration_ms', sa.Float(), nullable=False),
        sa.Column('worker', sa.String(length=200), nullable=True),
        sa.Column('attrs', sa.JSON(), nullable=True),
    )
    op.create_index('ix_stage_events_run', 'stage_events', ['run_id', 'attempt', 'seq'])
    op.create_index('ix_stage_events_started', 'stage_events', ['started_at'])
    op.create_index('ix_stage_events_stage_started', 'stage_events', ['stage', 'started_at'])
    op.create_table(
        'audio_features',
        sa.Column('audio_id', sa.String(length=32),
                  sa.ForeignKey('audio.id', name='fk_audio_features_audio_id', ondelete='CASCADE'),
                  primary_key=True),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('computed_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('duration_s', sa.Float(), nullable=False),
        sa.Column('rms_dbfs', sa.Float(), nullable=True),
        sa.Column('peak_dbfs', sa.Float(), nullable=True),
        sa.Column('clip_ratio', sa.Float(), nullable=True),
        sa.Column('speech_ratio', sa.Float(), nullable=True),
        sa.Column('n_speech_segments', sa.Integer(), nullable=True),
        sa.Column('snr_db', sa.Float(), nullable=True),
        sa.Column('spectral_centroid_hz', sa.Float(), nullable=True),
        sa.Column('n_chunks', sa.Integer(), nullable=True),
        sa.Column('padding_ratio', sa.Float(), nullable=True),
        sa.Column('extra', sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table('audio_features')
    op.drop_index('ix_stage_events_stage_started', table_name='stage_events')
    op.drop_index('ix_stage_events_started', table_name='stage_events')
    op.drop_index('ix_stage_events_run', table_name='stage_events')
    op.drop_table('stage_events')
