CREATE TABLE IF NOT EXISTS videos(
  vid_UUID INTEGER PRIMARY KEY
);


CREATE TABLE IF NOT EXISTS analysis(
  analysis_UUID INTEGER PRIMARY KEY,
  vid_UUID INTEGER NOT NULL,
  FOREIGN KEY(vid_UUID) REFERENCES videos(vid_UUID)
);



CREATE TABLE IF NOT EXISTS teams(
  team_UUID INTEGER PRIMARY KEY,
  team_side TEXT NOT NULL
);

INSERT INTO teams(team_UUID, team_side )
  VALUES(1, "left");
INSERT INTO teams(team_UUID, team_side )
  VALUES(2, "right");


CREATE TABLE IF NOT EXISTS players(
  player_UUID INTEGER PRIMARY KEY
);

INSERT INTO players(player_UUID )
  VALUES(1);
INSERT INTO players(player_UUID )
  VALUES(2);
INSERT INTO players(player_UUID )
  VALUES(3);
INSERT INTO players(player_UUID )
  VALUES(4);
INSERT INTO players(player_UUID )
  VALUES(5);
INSERT INTO players(player_UUID )
  VALUES(6);


CREATE TABLE IF NOT EXISTS death_states(
  death_state_UUID INTEGER PRIMARY KEY,
  death_state_name TEXT NOT NULL
);

INSERT INTO death_states(death_state_name )
  VALUES("DEATH");
INSERT INTO death_states(death_state_name )
  VALUES("RESPAWN");

  
CREATE TABLE IF NOT EXISTS ultimate_states(
  ultimate_state_UUID INTEGER PRIMARY KEY,
  ultimate_state_name TEXT NOT NULL
);

INSERT INTO ultimate_states(ultimate_state_name )
  VALUES("READY");
INSERT INTO ultimate_states(ultimate_state_name )
  VALUES("USED");
  

CREATE TABLE IF NOT EXISTS ability_keys(
  ability_key_UUID INTEGER PRIMARY KEY,
  ability_key_name TEXT NOT NULL
);


INSERT INTO ability_keys(ability_key_UUID, ability_key_name )
  VALUES(1, "SHIFT");
INSERT INTO ability_keys(ability_key_UUID, ability_key_name )
  VALUES(2, "E");
INSERT INTO ability_keys(ability_key_UUID, ability_key_name )
  VALUES(3, "ULTIMATE1");
INSERT INTO ability_keys(ability_key_UUID, ability_key_name )
  VALUES(4, "ULTIMATE2");
INSERT INTO ability_keys(ability_key_UUID, ability_key_name )
  VALUES(5, "RIGHT_CLICK");
INSERT INTO ability_keys(ability_key_UUID, ability_key_name )
  VALUES(6, "PASSIVE");


CREATE TABLE IF NOT EXISTS side_types(
  side_type_UUID INTEGER PRIMARY KEY,
  side_type_name TEXT NOT NULL
);

INSERT INTO side_types(side_type_name)
  VALUES("ATK / DEF");
INSERT INTO side_types(side_type_name)
  VALUES("DEF / ATK");
INSERT INTO side_types(side_type_name)
  VALUES("NO SIDE");


CREATE TABLE IF NOT EXISTS maps(
  map_UUID INTEGER PRIMARY KEY,
  map_name TEXT NOT NULL
);

INSERT INTO maps(map_name)
  VALUES("AYUTTHAYA");
INSERT INTO maps(map_name )
  VALUES("BLACK FOREST");
INSERT INTO maps(map_name )
  VALUES("BLIZZARD WORLD");
INSERT INTO maps(map_name )
  VALUES("BUSAN");
INSERT INTO maps(map_name )
  VALUES("CASTILLO");
INSERT INTO maps(map_name )
  VALUES("CHÂTEAU GUILLARD");
INSERT INTO maps(map_name )
  VALUES("DORADO");
INSERT INTO maps(map_name )
  VALUES("ECOPOINT: ANTARTICA");
INSERT INTO maps(map_name )
  VALUES("EICHENWALDE");
INSERT INTO maps(map_name )
  VALUES("HANAMURA");
INSERT INTO maps(map_name )
  VALUES("HAVANA");
INSERT INTO maps(map_name )
  VALUES("HOLLYWOOD");
INSERT INTO maps(map_name )
  VALUES("HORIZON LUNAR COLONY");
INSERT INTO maps(map_name )
  VALUES("ILIOS");
INSERT INTO maps(map_name )
  VALUES("JUNKERTOWN");
INSERT INTO maps(map_name )
  VALUES("KING'S ROW");
INSERT INTO maps(map_name )
  VALUES("LIJIANG TOWER");
INSERT INTO maps(map_name )
  VALUES("NECROPOLIS");
INSERT INTO maps(map_name )
  VALUES("NEPAL");
INSERT INTO maps(map_name )
  VALUES("NUMBANI");
INSERT INTO maps(map_name )
  VALUES("OASIS");
INSERT INTO maps(map_name )
  VALUES("PARIS");
INSERT INTO maps(map_name )
  VALUES("PETRA");
INSERT INTO maps(map_name )
  VALUES("RIALTO");
INSERT INTO maps(map_name )
  VALUES("ROUTE 66");
INSERT INTO maps(map_name )
  VALUES("TEMPLE OF ANUBIS");
INSERT INTO maps(map_name )
  VALUES("VOLSKAYA INDUSTRIES");
INSERT INTO maps(map_name )
  VALUES("WATCHPOINT: GIBRALTAR");


CREATE TABLE IF NOT EXISTS game_metadata(
  idx INTEGER PRIMARY KEY,
  vid_UUID INTEGER NOT NULL,
  frame_start INTEGER NOT NULL,
  frame_end INTEGER NOT NULL,
  map_name TEXT NOT NULL,
  side_type_name INTEGER NOT NULL,
  FOREIGN KEY(vid_UUID) REFERENCES videos(vid_UUID),
  FOREIGN KEY(map_name) REFERENCES maps(map_name),
  FOREIGN KEY(side_type_name) REFERENCES side_types(side_type_name)
);


CREATE TABLE IF NOT EXISTS team_colors(
  idx INTEGER PRIMARY KEY,
  vid_UUID INTEGER NOT NULL,
  team_UUID INTEGER NOT NULL,
  primary_hex TEXT NOT NULL,
  secondary_hex TEXT NOT NULL,
  FOREIGN KEY(vid_UUID) REFERENCES videos(vid_UUID),
  FOREIGN KEY(team_UUID) REFERENCES teams(team_UUID)
);

CREATE TABLE IF NOT EXISTS RAW_players_nicknames(
  idx INTEGER PRIMARY KEY,
  vid_UUID INTEGER NOT NULL,
  analysis_UUID INTEGER NOT NULL,
  frame INTEGER NOT NULL,
  player_UUID INTEGER NOT NULL,
  team_UUID INTEGER NOT NULL,
  player_name TEXT NOT NULL,
  FOREIGN KEY(vid_UUID) REFERENCES videos(vid_UUID),
  FOREIGN KEY(analysis_UUID) REFERENCES analysis(analysis_UUID),
  FOREIGN KEY(team_UUID) REFERENCES teams(team_UUID),
  FOREIGN KEY(player_UUID) REFERENCES players(player_UUID)
);


CREATE TABLE IF NOT EXISTS POST_players_nicknames(
  idx INTEGER PRIMARY KEY,
  vid_UUID INTEGER NOT NULL,
  analysis_UUID INTEGER NOT NULL,
  player_UUID INTEGER NOT NULL,
  team_UUID INTEGER NOT NULL,
  player_name TEXT NOT NULL,
  FOREIGN KEY(vid_UUID) REFERENCES videos(vid_UUID),
  FOREIGN KEY(analysis_UUID) REFERENCES analysis(analysis_UUID),
  FOREIGN KEY(team_UUID) REFERENCES teams(team_UUID),
  FOREIGN KEY(player_UUID) REFERENCES players(player_UUID)
);


CREATE TABLE IF NOT EXISTS VAL_players_nicknames(
  idx INTEGER PRIMARY KEY,
  vid_UUID INTEGER NOT NULL,
  analysis_UUID INTEGER NOT NULL,
  player_UUID INTEGER NOT NULL,
  team_UUID INTEGER NOT NULL,
  player_name TEXT NOT NULL,
  FOREIGN KEY(vid_UUID) REFERENCES videos(vid_UUID),
  FOREIGN KEY(analysis_UUID) REFERENCES analysis(analysis_UUID),
  FOREIGN KEY(team_UUID) REFERENCES teams(team_UUID),
  FOREIGN KEY(player_UUID) REFERENCES players(player_UUID)
);



CREATE TABLE IF NOT EXISTS RAW_killfeed(
  idx INTEGER PRIMARY KEY,
  vid_UUID INTEGER NOT NULL,
  analysis_UUID INTEGER NOT NULL,
  frame INTEGER NOT NULL,
  match_coordinates TEXT NOT NULL,
  is_crit INTEGER NOT NULL,
  killed_hero_name TEXT ,
  killed_hero_team_UUID INTEGER,
  killer_hero_name TEXT ,
  killer_hero_team_UUID INTEGER ,
  assists_names TEXT ,
  ability_key_UUID INTEGER ,
  FOREIGN KEY(vid_UUID) REFERENCES videos(vid_UUID),
  FOREIGN KEY(analysis_UUID) REFERENCES analysis(analysis_UUID),
  FOREIGN KEY(ability_key_UUID) REFERENCES ability_keys(ability_key_UUID),
  FOREIGN KEY(killed_hero_team_UUID) REFERENCES teams(team_UUID),
  FOREIGN KEY(killer_hero_team_UUID) REFERENCES teams(team_UUID)
);


CREATE TABLE IF NOT EXISTS POST_killfeed(
  idx INTEGER PRIMARY KEY,
  vid_UUID INTEGER NOT NULL,
  analysis_UUID INTEGER NOT NULL,
  frame INTEGER NOT NULL,
  match_coordinates TEXT NOT NULL,
  is_crit INTEGER NOT NULL,
  killed_hero_name TEXT ,
  killed_hero_team_UUID INTEGER,
  killer_hero_name TEXT ,
  killer_hero_team_UUID INTEGER ,
  assists_names TEXT ,
  ability_key_UUID INTEGER ,
  FOREIGN KEY(vid_UUID) REFERENCES videos(vid_UUID),
  FOREIGN KEY(analysis_UUID) REFERENCES analysis(analysis_UUID),
  FOREIGN KEY(ability_key_UUID) REFERENCES ability_keys(ability_key_UUID),
  FOREIGN KEY(killed_hero_team_UUID) REFERENCES teams(team_UUID),
  FOREIGN KEY(killer_hero_team_UUID) REFERENCES teams(team_UUID)
);

CREATE TABLE IF NOT EXISTS POST_EVENTS_killfeed(
  idx INTEGER PRIMARY KEY,
  vid_UUID INTEGER NOT NULL,
  analysis_UUID INTEGER NOT NULL,
  frame INTEGER NOT NULL,
  match_coordinates TEXT NOT NULL,
  is_crit INTEGER NOT NULL,
  killed_hero_name TEXT ,
  killed_hero_team_UUID INTEGER,
  killer_hero_name TEXT ,
  killer_hero_team_UUID INTEGER ,
  assists_names TEXT ,
  ability_key_UUID INTEGER ,
  FOREIGN KEY(vid_UUID) REFERENCES videos(vid_UUID),
  FOREIGN KEY(analysis_UUID) REFERENCES analysis(analysis_UUID),
  FOREIGN KEY(ability_key_UUID) REFERENCES ability_keys(ability_key_UUID),
  FOREIGN KEY(killed_hero_team_UUID) REFERENCES teams(team_UUID),
  FOREIGN KEY(killer_hero_team_UUID) REFERENCES teams(team_UUID)
);

CREATE TABLE IF NOT EXISTS VAL_EVENTS_killfeed(
  idx INTEGER PRIMARY KEY,
  vid_UUID INTEGER NOT NULL,
  analysis_UUID INTEGER NOT NULL,
  frame INTEGER NOT NULL,
  match_coordinates TEXT NOT NULL,
  is_crit INTEGER NOT NULL,
  killed_hero_name TEXT NOT NULL,
  killed_hero_team_UUID INTEGER NOT NULL,
  killer_hero_name TEXT ,
  killer_hero_team_UUID INTEGER ,
  assists_names TEXT ,
  ability_key_UUID INTEGER ,
  FOREIGN KEY(vid_UUID) REFERENCES videos(vid_UUID),
  FOREIGN KEY(analysis_UUID) REFERENCES analysis(analysis_UUID),
  FOREIGN KEY(ability_key_UUID) REFERENCES ability_keys(ability_key_UUID),
  FOREIGN KEY(killed_hero_team_UUID) REFERENCES teams(team_UUID),
  FOREIGN KEY(killer_hero_team_UUID) REFERENCES teams(team_UUID)
);



CREATE TABLE IF NOT EXISTS RAW_healthbars(
  idx INTEGER PRIMARY KEY,
  vid_UUID INTEGER NOT NULL,
  analysis_UUID INTEGER NOT NULL,
  frame INTEGER NOT NULL,
  team_UUID INTEGER NOT NULL,
  player_UUID INTEGER NOT NULL,
  health INTEGER NOT NULL,
  shield INTEGER NOT NULL,
  armor INTEGER NOT NULL,
  damage INTEGER NOT NULL,
  FOREIGN KEY(vid_UUID) REFERENCES videos(vid_UUID),
  FOREIGN KEY(analysis_UUID) REFERENCES analysis(analysis_UUID),
  FOREIGN KEY(team_UUID) REFERENCES teams(team_UUID),
  FOREIGN KEY(player_UUID) REFERENCES players(player_UUID)
);

CREATE TABLE IF NOT EXISTS POST_healthbars(
  idx INTEGER PRIMARY KEY,
  vid_UUID INTEGER NOT NULL,
  analysis_UUID INTEGER NOT NULL,
  frame INTEGER NOT NULL,
  team_UUID INTEGER NOT NULL,
  player_UUID INTEGER NOT NULL,
  health INTEGER NOT NULL,
  shield INTEGER NOT NULL,
  armor INTEGER NOT NULL,
  damage INTEGER NOT NULL,
  FOREIGN KEY(vid_UUID) REFERENCES videos(vid_UUID),
  FOREIGN KEY(analysis_UUID) REFERENCES analysis(analysis_UUID),
  FOREIGN KEY(team_UUID) REFERENCES teams(team_UUID),
  FOREIGN KEY(player_UUID) REFERENCES players(player_UUID)
);


CREATE TABLE IF NOT EXISTS POST_EVENTS_healthbars(
  idx INTEGER PRIMARY KEY,
  vid_UUID INTEGER NOT NULL,
  analysis_UUID INTEGER NOT NULL,
  frame INTEGER NOT NULL,
  team_UUID INTEGER NOT NULL,
  player_UUID INTEGER NOT NULL,
  death_state INTEGER NOT NULL,
  FOREIGN KEY(vid_UUID) REFERENCES videos(vid_UUID),
  FOREIGN KEY(analysis_UUID) REFERENCES analysis(analysis_UUID),
  FOREIGN KEY(death_state) REFERENCES death_states(death_state_UUID),
  FOREIGN KEY(team_UUID) REFERENCES teams(team_UUID),
  FOREIGN KEY(player_UUID) REFERENCES players(player_UUID)
);


CREATE TABLE IF NOT EXISTS VAL_EVENTS_healthbars(
  idx INTEGER PRIMARY KEY,
  vid_UUID INTEGER NOT NULL,
  analysis_UUID INTEGER NOT NULL,
  frame INTEGER NOT NULL,
  team_UUID INTEGER NOT NULL,
  player_UUID INTEGER NOT NULL,
  death_state INTEGER NOT NULL,
  FOREIGN KEY(vid_UUID) REFERENCES videos(vid_UUID),
  FOREIGN KEY(analysis_UUID) REFERENCES analysis(analysis_UUID),
  FOREIGN KEY(death_state) REFERENCES death_states(death_state_UUID),
  FOREIGN KEY(team_UUID) REFERENCES teams(team_UUID),
  FOREIGN KEY(player_UUID) REFERENCES players(player_UUID)
);




CREATE TABLE IF NOT EXISTS RAW_players_heroes(
  idx INTEGER PRIMARY KEY,
  vid_UUID INTEGER NOT NULL,
  analysis_UUID INTEGER NOT NULL,
  frame INTEGER NOT NULL,
  player_UUID INTEGER NOT NULL,
  team_UUID INTEGER NOT NULL,
  hero_name TEXT NOT NULL,
  FOREIGN KEY(vid_UUID) REFERENCES videos(vid_UUID),
  FOREIGN KEY(analysis_UUID) REFERENCES analysis(analysis_UUID),
  FOREIGN KEY(team_UUID) REFERENCES teams(team_UUID),
  FOREIGN KEY(player_UUID) REFERENCES players(player_UUID)
);


CREATE TABLE IF NOT EXISTS POST_players_heroes(
  idx INTEGER PRIMARY KEY,
  vid_UUID INTEGER NOT NULL,
  analysis_UUID INTEGER NOT NULL,
  frame INTEGER NOT NULL,
  player_UUID INTEGER NOT NULL,
  team_UUID INTEGER NOT NULL,
  hero_name TEXT NOT NULL,
  FOREIGN KEY(vid_UUID) REFERENCES videos(vid_UUID),
  FOREIGN KEY(analysis_UUID) REFERENCES analysis(analysis_UUID),
  FOREIGN KEY(team_UUID) REFERENCES teams(team_UUID),
  FOREIGN KEY(player_UUID) REFERENCES players(player_UUID)
);


CREATE TABLE IF NOT EXISTS POST_EVENTS_players_heroes(
  idx INTEGER PRIMARY KEY,
  vid_UUID INTEGER NOT NULL,
  analysis_UUID INTEGER NOT NULL,
  frame INTEGER NOT NULL,
  player_UUID INTEGER NOT NULL,
  team_UUID INTEGER NOT NULL,
  hero_name TEXT NOT NULL,
  FOREIGN KEY(vid_UUID) REFERENCES videos(vid_UUID),
  FOREIGN KEY(analysis_UUID) REFERENCES analysis(analysis_UUID),
  FOREIGN KEY(team_UUID) REFERENCES teams(team_UUID),
  FOREIGN KEY(player_UUID) REFERENCES players(player_UUID)
);


CREATE TABLE IF NOT EXISTS VAL_EVENTS_players_heroes(
  idx INTEGER PRIMARY KEY,
  vid_UUID INTEGER NOT NULL,
  analysis_UUID INTEGER NOT NULL,
  frame INTEGER NOT NULL,
  player_UUID INTEGER NOT NULL,
  team_UUID INTEGER NOT NULL,
  hero_name TEXT NOT NULL,
  FOREIGN KEY(vid_UUID) REFERENCES videos(vid_UUID),
  FOREIGN KEY(analysis_UUID) REFERENCES analysis(analysis_UUID),
  FOREIGN KEY(team_UUID) REFERENCES teams(team_UUID),
  FOREIGN KEY(player_UUID) REFERENCES players(player_UUID)
);



CREATE TABLE IF NOT EXISTS RAW_team_names(
  idx INTEGER PRIMARY KEY,
  vid_UUID INTEGER NOT NULL,
  analysis_UUID INTEGER NOT NULL,
  frame INTEGER NOT NULL,
  team_UUID INTEGER NOT NULL,
  team_name TEXT NOT NULL,
  FOREIGN KEY(vid_UUID) REFERENCES videos(vid_UUID),
  FOREIGN KEY(analysis_UUID) REFERENCES analysis(analysis_UUID),
  FOREIGN KEY(team_UUID) REFERENCES teams(team_UUID)
);


CREATE TABLE IF NOT EXISTS POST_team_names(
  idx INTEGER PRIMARY KEY,
  vid_UUID INTEGER NOT NULL,
  analysis_UUID INTEGER NOT NULL,
  team_UUID INTEGER NOT NULL,
  team_name TEXT NOT NULL,
  FOREIGN KEY(vid_UUID) REFERENCES videos(vid_UUID),
  FOREIGN KEY(analysis_UUID) REFERENCES analysis(analysis_UUID),
  FOREIGN KEY(team_UUID) REFERENCES teams(team_UUID)
);


CREATE TABLE IF NOT EXISTS VAL_team_names(
  idx INTEGER PRIMARY KEY,
  vid_UUID INTEGER NOT NULL,
  analysis_UUID INTEGER NOT NULL,
  team_UUID INTEGER NOT NULL,
  team_name TEXT NOT NULL,
  FOREIGN KEY(vid_UUID) REFERENCES videos(vid_UUID),
  FOREIGN KEY(analysis_UUID) REFERENCES analysis(analysis_UUID),
  FOREIGN KEY(team_UUID) REFERENCES teams(team_UUID)
);



CREATE TABLE IF NOT EXISTS RAW_ultimate(
  idx INTEGER PRIMARY KEY,
  vid_UUID INTEGER NOT NULL,
  analysis_UUID INTEGER NOT NULL,
  team_UUID INTEGER NOT NULL,
  player_UUID INTEGER NOT NULL,
  ultimate_state INTEGER NOT NULL,
  frame INTEGER NOT NULL,
  FOREIGN KEY(vid_UUID) REFERENCES videos(vid_UUID),
  FOREIGN KEY(analysis_UUID) REFERENCES analysis(analysis_UUID),
  FOREIGN KEY(team_UUID) REFERENCES teams(team_UUID),
  FOREIGN KEY(player_UUID) REFERENCES players(player_UUID)
);


CREATE TABLE IF NOT EXISTS POST_ultimate(
  idx INTEGER PRIMARY KEY,
  vid_UUID INTEGER NOT NULL,
  analysis_UUID INTEGER NOT NULL,
  team_UUID INTEGER NOT NULL,
  player_UUID INTEGER NOT NULL,
  ultimate_state INTEGER NOT NULL,
  frame INTEGER NOT NULL,
  FOREIGN KEY(vid_UUID) REFERENCES videos(vid_UUID),
  FOREIGN KEY(analysis_UUID) REFERENCES analysis(analysis_UUID),
  FOREIGN KEY(team_UUID) REFERENCES teams(team_UUID),
  FOREIGN KEY(player_UUID) REFERENCES players(player_UUID)
);


CREATE TABLE IF NOT EXISTS POST_EVENTS_ultimate(
  idx INTEGER PRIMARY KEY,
  vid_UUID INTEGER NOT NULL,
  analysis_UUID INTEGER NOT NULL,
  team_UUID INTEGER NOT NULL,
  player_UUID INTEGER NOT NULL,
  ultimate_state_name TEXT NOT NULL,
  frame INTEGER NOT NULL,
  FOREIGN KEY(vid_UUID) REFERENCES videos(vid_UUID),
  FOREIGN KEY(analysis_UUID) REFERENCES analysis(analysis_UUID),
  FOREIGN KEY(ultimate_state_name) REFERENCES ultimate_states(ultimate_state_name),
  FOREIGN KEY(team_UUID) REFERENCES teams(team_UUID),
  FOREIGN KEY(player_UUID) REFERENCES players(player_UUID)
);



CREATE TABLE IF NOT EXISTS VAL_EVENTS_ultimate(
  idx INTEGER PRIMARY KEY,
  vid_UUID INTEGER NOT NULL,
  analysis_UUID INTEGER NOT NULL,
  team_UUID INTEGER NOT NULL,
  player_UUID INTEGER NOT NULL,
  ultimate_state_name TEXT NOT NULL,
  frame INTEGER NOT NULL,
  FOREIGN KEY(vid_UUID) REFERENCES videos(vid_UUID),
  FOREIGN KEY(analysis_UUID) REFERENCES analysis(analysis_UUID),
  FOREIGN KEY(ultimate_state_name) REFERENCES ultimate_states(ultimate_state_name),
  FOREIGN KEY(team_UUID) REFERENCES teams(team_UUID),
  FOREIGN KEY(player_UUID) REFERENCES players(player_UUID)
);
