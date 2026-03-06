--
-- PostgreSQL database dump
--

\restrict OXhhhbnVKh2IhojNclibOiE2iQU2S2mEOX5O0F3LAO3g9PMcFkllUOTFLNkerjm

-- Dumped from database version 16.12 (Homebrew)
-- Dumped by pg_dump version 18.3

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: alembic_version; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.alembic_version (
    version_num character varying(32) NOT NULL
);


--
-- Name: event_closing_lines; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.event_closing_lines (
    id integer NOT NULL,
    event_id character varying(128) NOT NULL,
    sport character varying(64) NOT NULL,
    market character varying(32) NOT NULL,
    selection character varying(256) NOT NULL,
    home_team character varying(256),
    away_team character varying(256),
    sharp_book character varying(64) NOT NULL,
    closing_odds double precision NOT NULL,
    closing_implied_prob double precision NOT NULL,
    closing_vig double precision,
    model_prob_at_signal double precision,
    model_ev_at_signal double precision,
    commence_time timestamp with time zone,
    logged_at timestamp with time zone NOT NULL
);


--
-- Name: event_closing_lines_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.event_closing_lines_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: event_closing_lines_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.event_closing_lines_id_seq OWNED BY public.event_closing_lines.id;


--
-- Name: event_stats_snapshots; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.event_stats_snapshots (
    id integer NOT NULL,
    event_id character varying(128) NOT NULL,
    sport character varying(64) NOT NULL,
    team character varying(256) NOT NULL,
    is_home boolean NOT NULL,
    matches_played integer,
    wins integer,
    draws integer,
    losses integer,
    goals_scored_avg double precision,
    goals_conceded_avg double precision,
    clean_sheets integer,
    attack_strength double precision,
    defense_strength double precision,
    form_trend_slope double precision,
    over25_rate double precision,
    btts_rate double precision,
    rest_days integer,
    schedule_congestion double precision,
    home_win_rate double precision,
    away_win_rate double precision,
    home_goals_avg double precision,
    away_goals_avg double precision,
    league_position integer,
    opponent_league_position integer,
    snapshot_at timestamp with time zone NOT NULL
);


--
-- Name: event_stats_snapshots_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.event_stats_snapshots_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: event_stats_snapshots_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.event_stats_snapshots_id_seq OWNED BY public.event_stats_snapshots.id;


--
-- Name: placed_bets; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.placed_bets (
    id integer NOT NULL,
    event_id character varying(128) NOT NULL,
    sport character varying(64) NOT NULL,
    market character varying(32) NOT NULL,
    selection character varying(256) NOT NULL,
    odds double precision NOT NULL,
    odds_open double precision,
    odds_close double precision,
    clv double precision,
    stake double precision NOT NULL,
    status character varying(16) NOT NULL,
    pnl double precision,
    sharp_implied_prob double precision,
    sharp_vig double precision,
    sentiment_delta double precision,
    injury_delta double precision,
    form_winrate_l5 double precision,
    form_games_l5 double precision,
    meta_features jsonb,
    notes text,
    is_training_data boolean DEFAULT false NOT NULL,
    data_source character varying(32) DEFAULT 'live_trade'::character varying NOT NULL,
    owner_chat_id character varying(64),
    created_at timestamp with time zone NOT NULL,
    updated_at timestamp with time zone,
    operator_id character varying(64),
    confirmed_odds double precision,
    confirmed_stake double precision,
    human_action character varying(16),
    override_reason text,
    reviewed_at timestamp with time zone,
    sharp_closing_odds double precision,
    sharp_closing_prob double precision,
    commence_time timestamp with time zone
);


--
-- Name: placed_bets_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.placed_bets_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: placed_bets_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.placed_bets_id_seq OWNED BY public.placed_bets.id;


--
-- Name: team_match_stats; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.team_match_stats (
    id integer NOT NULL,
    source_match_id character varying(128) NOT NULL,
    sport character varying(64) NOT NULL,
    league character varying(128),
    season character varying(32),
    matchday integer,
    match_date timestamp with time zone NOT NULL,
    team character varying(256) NOT NULL,
    opponent character varying(256) NOT NULL,
    is_home boolean NOT NULL,
    goals_for integer,
    goals_against integer,
    result character varying(4),
    shots integer,
    shots_on_target integer,
    possession_pct double precision,
    corners integer,
    fouls integer,
    yellow_cards integer,
    red_cards integer,
    ht_goals_for integer,
    ht_goals_against integer,
    extra_stats jsonb,
    source character varying(64) NOT NULL,
    created_at timestamp with time zone NOT NULL
);


--
-- Name: team_match_stats_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.team_match_stats_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: team_match_stats_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.team_match_stats_id_seq OWNED BY public.team_match_stats.id;


--
-- Name: event_closing_lines id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.event_closing_lines ALTER COLUMN id SET DEFAULT nextval('public.event_closing_lines_id_seq'::regclass);


--
-- Name: event_stats_snapshots id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.event_stats_snapshots ALTER COLUMN id SET DEFAULT nextval('public.event_stats_snapshots_id_seq'::regclass);


--
-- Name: placed_bets id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.placed_bets ALTER COLUMN id SET DEFAULT nextval('public.placed_bets_id_seq'::regclass);


--
-- Name: team_match_stats id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.team_match_stats ALTER COLUMN id SET DEFAULT nextval('public.team_match_stats_id_seq'::regclass);


--
-- Name: alembic_version alembic_version_pkc; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.alembic_version
    ADD CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num);


--
-- Name: event_closing_lines event_closing_lines_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.event_closing_lines
    ADD CONSTRAINT event_closing_lines_pkey PRIMARY KEY (id);


--
-- Name: event_stats_snapshots event_stats_snapshots_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.event_stats_snapshots
    ADD CONSTRAINT event_stats_snapshots_pkey PRIMARY KEY (id);


--
-- Name: placed_bets placed_bets_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.placed_bets
    ADD CONSTRAINT placed_bets_pkey PRIMARY KEY (id);


--
-- Name: team_match_stats team_match_stats_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.team_match_stats
    ADD CONSTRAINT team_match_stats_pkey PRIMARY KEY (id);


--
-- Name: event_closing_lines uq_closing_event_sel_mkt; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.event_closing_lines
    ADD CONSTRAINT uq_closing_event_sel_mkt UNIQUE (event_id, selection, market);


--
-- Name: placed_bets uq_event_sel_market_owner_source; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.placed_bets
    ADD CONSTRAINT uq_event_sel_market_owner_source UNIQUE (event_id, selection, market, owner_chat_id, data_source);


--
-- Name: event_stats_snapshots uq_event_team_snapshot; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.event_stats_snapshots
    ADD CONSTRAINT uq_event_team_snapshot UNIQUE (event_id, team);


--
-- Name: team_match_stats uq_match_team_source; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.team_match_stats
    ADD CONSTRAINT uq_match_team_source UNIQUE (source_match_id, team, source);


--
-- Name: ix_event_closing_lines_event_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_event_closing_lines_event_id ON public.event_closing_lines USING btree (event_id);


--
-- Name: ix_event_stats_snapshots_event_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_event_stats_snapshots_event_id ON public.event_stats_snapshots USING btree (event_id);


--
-- Name: ix_placed_bets_event_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_placed_bets_event_id ON public.placed_bets USING btree (event_id);


--
-- Name: ix_placed_bets_owner_chat_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_placed_bets_owner_chat_id ON public.placed_bets USING btree (owner_chat_id);


--
-- Name: ix_team_match_stats_match_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_team_match_stats_match_date ON public.team_match_stats USING btree (match_date);


--
-- Name: ix_team_match_stats_source_match_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_team_match_stats_source_match_id ON public.team_match_stats USING btree (source_match_id);


--
-- Name: ix_tms_sport_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tms_sport_date ON public.team_match_stats USING btree (sport, match_date);


--
-- Name: ix_tms_team_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tms_team_date ON public.team_match_stats USING btree (team, match_date);


--
-- PostgreSQL database dump complete
--

\unrestrict OXhhhbnVKh2IhojNclibOiE2iQU2S2mEOX5O0F3LAO3g9PMcFkllUOTFLNkerjm

