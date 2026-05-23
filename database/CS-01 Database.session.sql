-- 1. ล้างตารางเก่าเพื่อโครงสร้างใหม่ที่ถูกต้อง (ระวัง! ข้อมูลเก่าจะหายไป)
DROP TABLE IF EXISTS portfolio_assets CASCADE;
DROP TABLE IF EXISTS portfolios CASCADE;
DROP TABLE IF EXISTS stock_views CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- 2. ตาราง Users: เก็บข้อมูลอ้างอิงผู้ใช้จาก Clerk
CREATE TABLE users (
    clerk_id VARCHAR(255) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. ตาราง Stock Views: เก็บมุมมองผลตอบแทนล่วงหน้า (ใช้ใน Black-Litterman Model)
CREATE TABLE stock_views (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(50) UNIQUE NOT NULL,
    expected_return NUMERIC(8, 4) NOT NULL,
    variance NUMERIC(8, 4) DEFAULT 0.02,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. ตาราง Portfolios: เก็บข้อมูลพอร์ตโฟลิโอหลักที่ AI สร้างขึ้น
CREATE TABLE portfolios (
    id SERIAL PRIMARY KEY,
    clerk_id VARCHAR(255) NOT NULL REFERENCES users(clerk_id) ON DELETE CASCADE,
    target_beta NUMERIC(5, 2),
    budget NUMERIC(15, 2),
    target_amount NUMERIC(15, 2),
    duration_years INT,
    expected_return NUMERIC(8, 4),
    portfolio_volatility NUMERIC(8, 4),
    success_probability NUMERIC(5, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. ตาราง Portfolio Assets: แตกข้อมูลหุ้นรายตัวออกมาจาก JSONB (Normalization - 1st Normal Form)
CREATE TABLE portfolio_assets (
    id SERIAL PRIMARY KEY,
    portfolio_id INT NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE, -- เปลี่ยนตรงนี้เป็น INT
    ticker VARCHAR(50) NOT NULL,
    weight NUMERIC(5, 4) NOT NULL,
    beta NUMERIC(5, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. สร้าง Index เพื่อความเร็วในการค้นหา
CREATE INDEX idx_portfolios_clerk_id ON portfolios(clerk_id);
CREATE INDEX idx_portfolio_assets_portfolio_id ON portfolio_assets(portfolio_id);