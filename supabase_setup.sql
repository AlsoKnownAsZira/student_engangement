-- ═══════════════════════════════════════════════════════════════════
-- Supabase SQL setup
-- Run this in your Supabase SQL Editor (Dashboard → SQL → New query)
-- ═══════════════════════════════════════════════════════════════════

-- 1. Create analyses table
CREATE TABLE IF NOT EXISTS analyses (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    original_filename TEXT NOT NULL,
    input_video_path  TEXT,
    output_video_path TEXT,
    csv_path          TEXT,
    status            TEXT NOT NULL DEFAULT 'uploading'
                      CHECK (status IN ('uploading','processing','completed','failed')),
    error_message     TEXT,
    total_frames      INT,
    total_students    INT,
    avg_engagement_score FLOAT,
    engagement_distribution JSONB,
    processing_time_seconds FLOAT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);

-- 2. Create student_results table
CREATE TABLE IF NOT EXISTS student_results (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id       UUID NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    track_id          INT NOT NULL,
    final_engagement  TEXT NOT NULL,
    engaged_votes     INT DEFAULT 0,
    moderate_votes    INT DEFAULT 0,
    disengaged_votes  INT DEFAULT 0,
    total_frames      INT DEFAULT 0,
    avg_confidence    FLOAT DEFAULT 0,
    vote_percentage   FLOAT DEFAULT 0
);

-- 3. Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_analyses_user_id ON analyses(user_id);
CREATE INDEX IF NOT EXISTS idx_analyses_status ON analyses(status);
CREATE INDEX IF NOT EXISTS idx_student_results_analysis ON student_results(analysis_id);

-- 4. Enable Row Level Security
ALTER TABLE analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE student_results ENABLE ROW LEVEL SECURITY;

-- 5. RLS policies — users can only access their own data

-- analyses: SELECT
CREATE POLICY "Users can view own analyses"
    ON analyses FOR SELECT
    USING (auth.uid() = user_id);

-- analyses: INSERT
CREATE POLICY "Users can insert own analyses"
    ON analyses FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- analyses: UPDATE
CREATE POLICY "Users can update own analyses"
    ON analyses FOR UPDATE
    USING (auth.uid() = user_id);

-- analyses: DELETE
CREATE POLICY "Users can delete own analyses"
    ON analyses FOR DELETE
    USING (auth.uid() = user_id);

-- student_results: SELECT (via analysis ownership)
CREATE POLICY "Users can view own student results"
    ON student_results FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM analyses
            WHERE analyses.id = student_results.analysis_id
              AND analyses.user_id = auth.uid()
        )
    );

-- student_results: INSERT (service role bypasses RLS, but add policy for safety)
CREATE POLICY "Service can insert student results"
    ON student_results FOR INSERT
    WITH CHECK (true);

-- student_results: DELETE (via analysis ownership)
CREATE POLICY "Users can delete own student results"
    ON student_results FOR DELETE
    USING (
        EXISTS (
            SELECT 1 FROM analyses
            WHERE analyses.id = student_results.analysis_id
              AND analyses.user_id = auth.uid()
        )
    );

-- ═══════════════════════════════════════════════════════════════════
-- 6. Storage buckets (run these MANUALLY in Supabase Dashboard)
-- ═══════════════════════════════════════════════════════════════════
-- Go to Storage → Create bucket:
--   Name: input-videos    (Private)
--   Name: output-videos   (Private)
--
-- Then add RLS policies in Storage → Policies for each bucket:
--   - Allow authenticated users to upload to their own folder
--   - Allow authenticated users to read from their own folder
-- ═══════════════════════════════════════════════════════════════════
