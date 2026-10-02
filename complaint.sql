USE smart_panchayat;

CREATE TABLE complaints (
    id INT AUTO_INCREMENT PRIMARY KEY,

    user_id INT NOT NULL,
    panchayat_id INT NULL,

    category ENUM(
        'Water',
        'Electricity',
        'Road',
        'Sanitation',
        'Street Light',
        'Drainage',
        'Other'
    ) NOT NULL,

    title VARCHAR(150) NOT NULL,

    description TEXT NOT NULL,

    priority ENUM(
        'High',
        'Medium',
        'Low'
    ) DEFAULT 'Medium',

    status ENUM(
        'Submitted',
        'Under Review',
        'Assigned',
        'In Progress',
        'Resolved',
        'Rejected'
    ) DEFAULT 'Submitted',

    officer_remark TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE CASCADE,

    FOREIGN KEY (panchayat_id)
        REFERENCES panchayats(id)
        ON DELETE SET NULL
);


CREATE TABLE complaint_updates (
    id INT AUTO_INCREMENT PRIMARY KEY,

    complaint_id INT NOT NULL,

    status VARCHAR(50) NOT NULL,

    message VARCHAR(255),

    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (complaint_id)
        REFERENCES complaints(id)
        ON DELETE CASCADE
);


CREATE TABLE notifications (
    id INT AUTO_INCREMENT PRIMARY KEY,

    user_id INT NOT NULL,

    complaint_id INT NOT NULL,

    message VARCHAR(255) NOT NULL,

    is_read BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE CASCADE,

    FOREIGN KEY (complaint_id)
        REFERENCES complaints(id)
        ON DELETE CASCADE
);


CREATE TABLE IF NOT EXISTS complaint_images (
    id INT AUTO_INCREMENT PRIMARY KEY,

    complaint_id INT NOT NULL,

    stored_filename VARCHAR(255) NOT NULL,

    content_type VARCHAR(100),

    detected_label VARCHAR(255),

    confidence FLOAT,

    is_low_confidence BOOLEAN DEFAULT FALSE,

    ai_suggested_category VARCHAR(100),

    ai_suggestion_confirmed BOOLEAN DEFAULT FALSE,

    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (complaint_id)
        REFERENCES complaints(id)
        ON DELETE CASCADE
);


CREATE TABLE IF NOT EXISTS complaint_audio (
    id INT AUTO_INCREMENT PRIMARY KEY,

    complaint_id INT NOT NULL,

    stored_filename VARCHAR(255) NOT NULL,

    content_type VARCHAR(100),

    duration_seconds FLOAT,

    language_selected VARCHAR(20),

    transcription_engine VARCHAR(100),

    raw_transcript TEXT,

    confirmed_transcript TEXT,

    asr_confidence FLOAT,

    is_low_confidence BOOLEAN DEFAULT FALSE,

    transcription_available BOOLEAN DEFAULT FALSE,

    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (complaint_id)
        REFERENCES complaints(id)
        ON DELETE CASCADE
);