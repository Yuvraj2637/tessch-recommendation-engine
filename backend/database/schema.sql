-- =============================================================
-- TESSCH Recommendation Engine — SQLite reference schema
-- =============================================================
-- GENERATED FILE — do not edit by hand.
--
-- `backend/database/models.py` is the single source of truth for the schema.
-- This file is a human-readable dump of what those models create, kept in the
-- repo so the table layout can be reviewed without running any Python.
--
-- Regenerate with:   python -m scripts.dump_schema
-- Create the tables: python -m scripts.init_db
-- =============================================================


CREATE TABLE products (
	product_id VARCHAR(10) NOT NULL, 
	product_name VARCHAR(200) NOT NULL, 
	category VARCHAR(50) NOT NULL, 
	upper_type VARCHAR(50), 
	color VARCHAR(50), 
	color_family VARCHAR(20), 
	style VARCHAR(50), 
	price FLOAT NOT NULL, 
	material VARCHAR(100), 
	collection VARCHAR(50), 
	description TEXT, 
	availability BOOLEAN NOT NULL, 
	image_url VARCHAR(300), 
	is_core BOOLEAN NOT NULL, 
	created_at DATETIME NOT NULL, 
	updated_at DATETIME NOT NULL, 
	PRIMARY KEY (product_id)
);

CREATE INDEX ix_products_category ON products (category);

CREATE INDEX ix_products_collection ON products (collection);

CREATE INDEX ix_products_color_family ON products (color_family);

CREATE INDEX ix_products_is_core ON products (is_core);

CREATE INDEX ix_products_style ON products (style);



CREATE TABLE users (
	user_id VARCHAR(20) NOT NULL, 
	username VARCHAR(100) NOT NULL, 
	email VARCHAR(200), 
	age_group VARCHAR(20), 
	gender VARCHAR(20), 
	style_preference VARCHAR(50), 
	device_preference VARCHAR(20), 
	is_new_user BOOLEAN NOT NULL, 
	signup_at DATETIME, 
	created_at DATETIME NOT NULL, 
	last_active_at DATETIME, 
	PRIMARY KEY (user_id), 
	UNIQUE (username), 
	UNIQUE (email)
);



CREATE TABLE sessions (
	session_id VARCHAR(30) NOT NULL, 
	user_id VARCHAR(20) NOT NULL, 
	started_at DATETIME NOT NULL, 
	ended_at DATETIME, 
	duration_seconds INTEGER, 
	events_count INTEGER NOT NULL, 
	products_viewed INTEGER NOT NULL, 
	unique_products INTEGER NOT NULL, 
	repeat_views INTEGER NOT NULL, 
	wishlist_count INTEGER NOT NULL, 
	cart_count INTEGER NOT NULL, 
	purchase_count INTEGER NOT NULL, 
	search_count INTEGER NOT NULL, 
	device_type VARCHAR(20), 
	intent VARCHAR(30), 
	created_at DATETIME NOT NULL, 
	PRIMARY KEY (session_id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id)
);

CREATE INDEX ix_sessions_user_id ON sessions (user_id);



CREATE TABLE user_item_features (
	user_id VARCHAR(20) NOT NULL, 
	product_id VARCHAR(10) NOT NULL, 
	impression_count INTEGER NOT NULL, 
	total_views INTEGER NOT NULL, 
	total_time_spent FLOAT NOT NULL, 
	avg_dwell_time FLOAT NOT NULL, 
	max_dwell_time FLOAT NOT NULL, 
	repeat_view_count INTEGER NOT NULL, 
	distinct_sessions INTEGER NOT NULL, 
	hover_count INTEGER NOT NULL, 
	click_count INTEGER NOT NULL, 
	wishlist_count INTEGER NOT NULL, 
	cart_count INTEGER NOT NULL, 
	purchase_count INTEGER NOT NULL, 
	share_count INTEGER NOT NULL, 
	days_since_last_view FLOAT, 
	days_since_last_interaction FLOAT, 
	days_since_last_purchase FLOAT, 
	first_interaction_at DATETIME, 
	last_interaction_at DATETIME, 
	raw_interest_score FLOAT NOT NULL, 
	time_decayed_score FLOAT NOT NULL, 
	target INTEGER NOT NULL, 
	computed_at DATETIME NOT NULL, 
	PRIMARY KEY (user_id, product_id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(product_id) REFERENCES products (product_id)
);

CREATE INDEX idx_uif_product_id ON user_item_features (product_id);

CREATE INDEX idx_uif_score ON user_item_features (time_decayed_score);

CREATE INDEX idx_uif_user_id ON user_item_features (user_id);



CREATE TABLE orders (
	order_id CHAR(32) NOT NULL, 
	user_id VARCHAR(20) NOT NULL, 
	session_id VARCHAR(30), 
	product_id VARCHAR(10) NOT NULL, 
	quantity INTEGER NOT NULL, 
	unit_price FLOAT NOT NULL, 
	total_amount FLOAT NOT NULL, 
	status VARCHAR(20) NOT NULL, 
	ordered_at DATETIME NOT NULL, 
	PRIMARY KEY (order_id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(session_id) REFERENCES sessions (session_id), 
	FOREIGN KEY(product_id) REFERENCES products (product_id)
);

CREATE INDEX ix_orders_product_id ON orders (product_id);

CREATE INDEX ix_orders_user_id ON orders (user_id);



CREATE TABLE recommendation_logs (
	log_id CHAR(32) NOT NULL, 
	user_id VARCHAR(20) NOT NULL, 
	product_id VARCHAR(10) NOT NULL, 
	session_id VARCHAR(30), 
	recommendation_rank INTEGER NOT NULL, 
	score FLOAT, 
	model_name VARCHAR(50) NOT NULL, 
	model_version VARCHAR(20) NOT NULL, 
	experiment_group VARCHAR(20), 
	was_clicked BOOLEAN NOT NULL, 
	was_carted BOOLEAN NOT NULL, 
	was_purchased BOOLEAN NOT NULL, 
	served_at DATETIME NOT NULL, 
	interacted_at DATETIME, 
	PRIMARY KEY (log_id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(product_id) REFERENCES products (product_id), 
	FOREIGN KEY(session_id) REFERENCES sessions (session_id)
);

CREATE INDEX idx_rec_logs_product_id ON recommendation_logs (product_id);

CREATE INDEX idx_rec_logs_served_at ON recommendation_logs (served_at);

CREATE INDEX idx_rec_logs_user_id ON recommendation_logs (user_id);



CREATE TABLE user_events (
	event_id CHAR(32) NOT NULL, 
	user_id VARCHAR(20) NOT NULL, 
	product_id VARCHAR(10), 
	session_id VARCHAR(30), 
	event_type VARCHAR(30) NOT NULL, 
	duration_seconds FLOAT, 
	surface VARCHAR(30), 
	position INTEGER, 
	search_query VARCHAR(300), 
	event_value FLOAT, 
	source VARCHAR(12) NOT NULL, 
	timestamp DATETIME NOT NULL, 
	metadata JSON, 
	PRIMARY KEY (event_id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(product_id) REFERENCES products (product_id), 
	FOREIGN KEY(session_id) REFERENCES sessions (session_id)
);

CREATE INDEX idx_user_events_event_type ON user_events (event_type);

CREATE INDEX idx_user_events_product_id ON user_events (product_id);

CREATE INDEX idx_user_events_session_id ON user_events (session_id);

CREATE INDEX idx_user_events_timestamp ON user_events (timestamp);

CREATE INDEX idx_user_events_user_id ON user_events (user_id);

CREATE INDEX idx_user_events_user_product ON user_events (user_id, product_id);

CREATE INDEX ix_user_events_source ON user_events (source);
