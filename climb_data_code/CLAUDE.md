# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a data analysis project that scrapes and analyzes boulder climbing data from Mountain Project, with a focus on gender-based rating statistics. The primary research questions are:
- For each boulder problem, display ratings broken down by overall, male, female, and undetermined gender
- Analyze how accurately gender can be inferred from user names (using `gender_guesser`)
- Extract and analyze comments from women climbers

## Environment Setup

The project uses a conda environment named `climb_data_env`:

```bash
conda activate climb_data_env
jupyter notebook 20260105_boulder_gender_stats.ipynb
```

Key dependencies: `pandas`, `requests`, `beautifulsoup4`, `gender_guesser`

## Architecture

The project is a single notebook (`20260105_boulder_gender_stats.ipynb`) implementing a data pipeline:

**Data Collection** → Mountain Project API v2 (`/api/v2/routes/`) and web scraping of user profile pages

**Core Functions:**
- `retrieve_overview_csv(URL)` — fetches the Mountain Project Route Finder CSV export for a given area URL; parses grades/danger ratings into a DataFrame
- `get_route_stats(URL, stat_type, params)` — calls the MP API v2 to retrieve stars, ratings, ticks, or todos for a route
- `compile_all_user_dict(star_df, rating_df, tick_df)` — merges user data across three DataFrames to build a deduplicated user dictionary
- `get_user_info(user_dict)` — scrapes each user's MP profile page to populate location, age, and listed gender
- `create_route_stats_df(route_name, ...)` — combines all stats for a single route into one DataFrame
- `fill_area_df_with_stats(area_df)` — iterates over all routes in an area and applies the above

**Data Flow:**
```
Mountain Project Route Finder CSV
  → per-route API calls (stars, ratings, ticks)
  → compile_all_user_dict → unique user list
  → get_user_info → demographics (location, age, listed_gender)
  → gender_guesser on user names → guessed_gender
  → per-route stats filtered by gender
```

**User dict schema:** `{user_ID, user_name, user_URL, location, age, listed_gender, guessed_gender}`

## Notes

- The MP API and profile scraping are rate-sensitive; cells that iterate over many routes/users can be slow
- `gender_guesser` is noted to have bias toward European names — accuracy on non-European names is a known limitation
- The related environment file `~/20240111_rec_env.yml` contains broader dependency definitions
