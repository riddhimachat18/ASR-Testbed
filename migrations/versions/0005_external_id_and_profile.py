"""audio.external_id and transcription_runs.profile

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-10

* audio.external_id: the file's ID in its source dataset (LibriSpeech utterance ID), unique per dataset, so runs
  can be exported and compared across databases. Backfilled by re-running scripts/import_librispeech.py.
* transcription_runs.profile: per-stage timings, cache hits and runtime details. Separate from `metrics` because
  rescoring rewrites `metrics`.
"""
from alembic import op
import sqlalchemy as sa


revision = '0005'
down_revision = '0004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('audio', schema=None) as batch_op:
        batch_op.add_column(sa.Column('external_id', sa.String(length=200), nullable=True))
        batch_op.create_index('ix_audio_dataset_external_id', ['dataset_id', 'external_id'], unique=True)
    with op.batch_alter_table('transcription_runs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('profile', sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('transcription_runs', schema=None) as batch_op:
        batch_op.drop_column('profile')
    with op.batch_alter_table('audio', schema=None) as batch_op:
        batch_op.drop_index('ix_audio_dataset_external_id')
        batch_op.drop_column('external_id')
