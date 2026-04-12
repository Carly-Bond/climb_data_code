import copy
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os
import pandas as pd
import re


COLOR_F = '#e06c8b'
COLOR_M = '#5b8db8'
COLOR_OTHER = '#a89cc8'

GUESS_MAP = {
    'male': 'Male', 'mostly_male': 'Male',
    'female': 'Female', 'mostly_female': 'Female',
    'andy': 'Unknown', 'unknown': 'Unknown',
}

GENDER_COLOR_MAP = {
    'male': COLOR_M, 'mostly_male': COLOR_M,
    'female': COLOR_F, 'mostly_female': COLOR_F,
}

# define functions for data analysis

def find_area_csvs(area_name, data_dir='data'):
    """
    Searches data_dir for CSV files whose names contain area_name.
    Groups matches by file type (the suffix after area_name in the filename),
    reads the most recent file of each type into a pandas DataFrame, and
    returns a dict mapping file type suffix to DataFrame.

    Args:
        area_name: The area name string to search for (e.g. AREA_NAME). Type is string.
        data_dir:  Subdirectory to search. Default is 'data'.

    Returns:
        dict mapping file type suffix (str) to pandas DataFrame.
        Example: {'area_df_with_stats': <DataFrame>, 'user_info_df': <DataFrame>}
    """
    pattern = os.path.join(data_dir, f'*{area_name}*.csv')
    matches = glob.glob(pattern)

    if not matches:
        print(f"No CSV files found in '{data_dir}' containing '{area_name}'.")
        return {}

    # Group by the suffix that follows area_name in the filename
    groups = {}
    for path in matches:
        filename = os.path.basename(path)
        after = filename.split(area_name, 1)[-1].lstrip('_').removesuffix('.csv')
        groups.setdefault(after, []).append(path)

    # Read the most recent file per type into a DataFrame
    result = {}
    for suffix, paths in groups.items():
        most_recent = sorted(paths)[-1]
        result[suffix] = pd.read_csv(most_recent)
        print(f"Loaded '{suffix}' from {most_recent}  ({len(result[suffix])} rows)")

    return result

def _derive_has_listed(user_info_df):
    """Return a copy of user_info_df filtered to rows with both listed and guessed
    gender, with guessed_normalized and match columns added."""
    has_listed = user_info_df.dropna(subset=['listed_gender', 'guessed_gender']).copy()
    has_listed['guessed_normalized'] = has_listed['guessed_gender'].map(GUESS_MAP).fillna('Unknown')
    has_listed['match'] = has_listed['listed_gender'] == has_listed['guessed_normalized']
    return has_listed


def plot_gender_breakdown(user_info_df):
    """Plot listed gender, guessed gender, and guessed-vs-listed match rate.

    Args:
        user_info_df: DataFrame with 'listed_gender' and 'guessed_gender' columns,
                      one row per user.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Plot 1: listed_gender breakdown ---
    listed_counts = user_info_df['listed_gender'].fillna('Unknown').value_counts()
    axes[0].bar(listed_counts.index, listed_counts.values)
    axes[0].set_title('Listed Gender Breakdown')
    axes[0].set_xlabel('listed_gender')
    axes[0].set_ylabel('# users')
    for i, v in enumerate(listed_counts.values):
        axes[0].text(i, v + 0.3, str(v), ha='center', fontweight='bold')

    # --- Plot 2: guessed_gender breakdown ---
    guessed_counts = user_info_df['guessed_gender'].fillna('unknown').value_counts()
    axes[1].bar(guessed_counts.index, guessed_counts.values)
    axes[1].set_title('Guessed Gender Breakdown\n(gender_guesser)')
    axes[1].set_xlabel('guessed_gender')
    axes[1].set_ylabel('# users')
    axes[1].tick_params(axis='x', rotation=30)
    for i, v in enumerate(guessed_counts.values):
        axes[1].text(i, v + 0.3, str(v), ha='center', fontweight='bold')

    # --- Plot 3: guessed vs listed match rate ---
    has_listed = _derive_has_listed(user_info_df)
    match_counts = has_listed['match'].value_counts()
    match_labels = ['Match' if k else 'Mismatch' for k in match_counts.index]
    axes[2].bar(match_labels, match_counts.values, color=['steelblue', 'salmon'])
    axes[2].set_title(
        f'Guessed vs Listed Gender Match\n(n={len(has_listed)} users with listed gender)'
    )
    axes[2].set_ylabel('# users')
    for i, v in enumerate(match_counts.values):
        axes[2].text(i, v + 0.3, str(v), ha='center', fontweight='bold')
    match_rate = has_listed['match'].mean()
    axes[2].set_xlabel(f'Match rate: {match_rate:.1%}')

    plt.tight_layout()
    plt.show()


def plot_gender_mismatch_detail(user_info_df):
    """Four-panel deep dive into gender mismatch patterns.

    Args:
        user_info_df: DataFrame with 'listed_gender', 'guessed_gender', and
                      'user_name' columns, one row per user.
    """
    has_listed = _derive_has_listed(user_info_df)
    mismatches = has_listed[~has_listed['match']].copy()
    n_mismatch = len(mismatches)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        f'Gender Mismatch Deep Dive  '
        f'(n={n_mismatch} mismatches out of {len(has_listed)} users with listed gender)',
        fontsize=13, fontweight='bold', y=1.01,
    )

    # ── Subplot 1 (top-left): Mismatch count stacked by listed_gender ──────────
    ax1 = axes[0, 0]
    pivot = (
        mismatches
        .groupby(['guessed_gender', 'listed_gender'])
        .size()
        .unstack(fill_value=0)
    )
    for col in ['Female', 'Male']:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot.sort_values('Male', ascending=True)

    bar_positions = list(range(len(pivot)))
    ax1.barh(bar_positions, pivot['Female'], color=COLOR_F, label='Listed Female')
    ax1.barh(bar_positions, pivot['Male'], left=pivot['Female'], color=COLOR_M, label='Listed Male')
    ax1.set_yticks(bar_positions)
    ax1.set_yticklabels(pivot.index)
    ax1.set_xlabel('# mismatched users')
    ax1.set_title('Mismatch Count by Guessed Category\n(stacked by listed gender)')
    ax1.legend(loc='lower right')
    for i, (f_val, m_val) in enumerate(zip(pivot['Female'], pivot['Male'])):
        ax1.text(f_val + m_val + 0.3, i, str(f_val + m_val), va='center', fontsize=9)

    # ── Subplot 2 (top-right): Active gender confusion ──────────────────────────
    ax2 = axes[0, 1]
    active_mis = mismatches[mismatches['guessed_normalized'] != 'Unknown']
    f_to_m = active_mis[active_mis['listed_gender'] == 'Female'].groupby('guessed_gender').size()
    m_to_f = active_mis[active_mis['listed_gender'] == 'Male'].groupby('guessed_gender').size()

    confusion_df = pd.DataFrame({
        'F listed\n→ guessed Male': f_to_m,
        'M listed\n→ guessed Female': m_to_f,
    }).fillna(0).astype(int)

    x = list(range(len(confusion_df)))
    width = 0.35
    for j, (col, color) in enumerate(zip(confusion_df.columns, [COLOR_F, COLOR_M])):
        bars = ax2.bar(
            [xi + j * width for xi in x],
            confusion_df[col],
            width=width, label=col, color=color, alpha=0.85,
        )
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.2,
                         str(int(h)), ha='center', va='bottom', fontsize=9)

    ax2.set_xticks([xi + width / 2 for xi in x])
    ax2.set_xticklabels(confusion_df.index, rotation=20, ha='right')
    ax2.set_ylabel('# mismatched users')
    ax2.set_title('Active Gender Confusion\n(excludes unknown/andy — guesser picked wrong binary)')
    ax2.legend()

    # ── Subplot 3 (bottom-left): Mismatch rate per guessed_gender category ──────
    ax3 = axes[1, 0]
    rate_data = []
    for cat, grp in has_listed.groupby('guessed_gender'):
        n_total = len(grp)
        n_mis = (~grp['match']).sum()
        rate_data.append({'guessed_gender': cat, 'n_total': n_total,
                          'n_mismatch': n_mis, 'rate': n_mis / n_total})
    rate_df = pd.DataFrame(rate_data).sort_values('rate', ascending=False)

    bar_colors = ['#c0392b' if r >= 1.0 else COLOR_M if r < 0.05 else COLOR_OTHER
                  for r in rate_df['rate']]
    bars3 = ax3.bar(rate_df['guessed_gender'], rate_df['rate'] * 100, color=bar_colors)
    ax3.axhline(100, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax3.set_ylim(0, 115)
    ax3.set_ylabel('Mismatch rate (%)')
    ax3.set_xlabel('guessed_gender')
    ax3.set_title('Mismatch Rate by Guessed Category\n(% of users with that guess who mismatch)')
    ax3.tick_params(axis='x', rotation=20)
    for bar, row in zip(bars3, rate_df.itertuples()):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f'{row.n_mismatch}/{row.n_total}',
            ha='center', va='bottom', fontsize=8.5,
        )

    # ── Subplot 4 (bottom-right): Example names text table ─────────────────────
    ax4 = axes[1, 1]
    ax4.axis('off')
    lines = ['Example user_names per mismatch group:\n']
    name_groups = (
        mismatches
        .groupby(['listed_gender', 'guessed_gender'])['user_name']
        .apply(lambda x: ', '.join(x.head(3)))
    )
    for (listed, guessed_raw), names in name_groups.items():
        n = len(mismatches[
            (mismatches['listed_gender'] == listed) &
            (mismatches['guessed_gender'] == guessed_raw)
        ])
        lines.append(f'Listed={listed}, Guessed={guessed_raw}  (n={n}):')
        lines.append(f'  {names}')
        lines.append('')
    ax4.text(
        0.03, 0.97, '\n'.join(lines),
        transform=ax4.transAxes,
        va='top', ha='left',
        fontsize=9,
        fontfamily='monospace',
    )
    ax4.set_title('Example Names per Mismatch Type', fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_top_users_by_ticks(area_df, top_n=200, gender_column='likely_gender'):
    """Bar chart of the top N users by tick count, colored by gender.

    Args:
        area_df:        DataFrame with one row per tick, including 'user.name' and
                        the gender column.
        top_n:          Number of top users to display. Default is 200.
        gender_column:  Column used to determine bar color. Default is 'likely_gender'.
    """
    vc = area_df['user.name'].value_counts()
    top_users = vc.head(top_n)

    user_gender = area_df.drop_duplicates('user.name').set_index('user.name')[gender_column]
    bar_colors = [GENDER_COLOR_MAP.get(user_gender.get(name), COLOR_OTHER)
                  for name in top_users.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(top_users)), top_users.values, color=bar_colors)
    ax.set_title(f'Top {top_n} users by tick count')
    ax.set_xlabel('users')
    ax.set_ylabel('count')

    legend_elements = [
        Patch(facecolor=COLOR_M, label='male / mostly_male'),
        Patch(facecolor=COLOR_F, label='female / mostly_female'),
        Patch(facecolor=COLOR_OTHER, label='other / unknown'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()

    print(f'Total unique users: {vc.shape[0]}')
    print(top_users.head(10))


def plot_group_vs_overall_rating(area_df, target_genders=None, min_counts=3,
                                 gender_column='likely_gender'):
    """Scatter + bar chart comparing a gender group's avg rating vs the overall avg.

    Args:
        area_df:        DataFrame with one row per rating, including 'Route', 'score',
                        and the gender column.
        target_genders: List of gender values to include in the group.
                        Default is ['female', 'mostly_female'].
        min_counts:     Minimum number of ratings required from both the target group
                        and male users for a route to be included. Default is 3.
        gender_column:  Column used to filter genders. Default is 'likely_gender'.
    """
    if target_genders is None:
        target_genders = ['female', 'mostly_female']
    male_genders = ['male', 'mostly_male']

    female_counts = (area_df[area_df[gender_column].isin(target_genders)]
                     .groupby('Route')['score'].count())
    male_counts = (area_df[area_df[gender_column].isin(male_genders)]
                   .groupby('Route')['score'].count())
    qualified_routes = (female_counts[female_counts >= min_counts].index
                        .intersection(male_counts[male_counts >= min_counts].index))
    print(f"{len(qualified_routes)} routes meet the minimum rating threshold "
          f"(>={min_counts} female, >={min_counts} male)")

    overall_avg = (area_df[area_df['Route'].isin(qualified_routes)]
                   .groupby('Route')['score'].mean()
                   .rename('overall_avg'))
    group_avg = (area_df[area_df[gender_column].isin(target_genders) &
                         area_df['Route'].isin(qualified_routes)]
                 .groupby('Route')['score'].mean()
                 .rename('group_avg'))

    compare_df = pd.concat([overall_avg, group_avg], axis=1).dropna()
    compare_df['diff'] = compare_df['group_avg'] - compare_df['overall_avg']
    compare_df = compare_df.sort_values('overall_avg')

    group_label = ' / '.join(target_genders)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: Scatter ---
    ax = axes[0]
    ax.scatter(compare_df['overall_avg'], compare_df['group_avg'], zorder=3)
    lims = [
        min(compare_df['overall_avg'].min(), compare_df['group_avg'].min()) - 0.1,
        max(compare_df['overall_avg'].max(), compare_df['group_avg'].max()) + 0.1,
    ]
    ax.plot(lims, lims, 'k--', linewidth=1, label='y = x (no difference)')
    for route, row in compare_df.iterrows():
        ax.annotate(route, (row['overall_avg'], row['group_avg']),
                    fontsize=7, textcoords='offset points', xytext=(4, 2))
    ax.set_xlabel('Overall avg star rating')
    ax.set_ylabel(f'{group_label} avg star rating')
    ax.set_title('Group vs Overall Star Rating per Route')
    ax.legend()

    # --- Plot 2: Difference bar chart ---
    ax = axes[1]
    colors = ['salmon' if d < 0 else 'steelblue' for d in compare_df['diff']]
    ax.barh(compare_df.index, compare_df['diff'], color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Difference (group avg − overall avg)')
    ax.set_title(f'Rating Difference: {group_label}\nvs Overall (per route)')
    ax.tick_params(axis='y', labelsize=8)

    plt.tight_layout()
    plt.show()

    print(compare_df[['overall_avg', 'group_avg', 'diff']].round(2))


def plot_female_vs_male_rating(area_df, min_counts=3, gender_column='likely_gender',
                               included_grades=None):
    """Scatter + bar chart comparing female vs male avg star rating per route.

    Args:
        area_df:          DataFrame with one row per rating, including 'Route', 'score',
                          'V-grade', and the gender column.
        min_counts:       Minimum number of ratings required from both female and male
                          users for a route to be included. Default is 3.
        gender_column:    Column used to filter genders. Default is 'likely_gender'.
        included_grades:  List of V-grades to include (e.g. ['V0', 'V1', 'V2']).
                          If None, all grades are included.
    """
    female_genders = ['female', 'mostly_female']
    male_genders = ['male', 'mostly_male']

    plot_df = area_df.copy()
    if included_grades is not None:
        # For each grade like 'V1', match:
        #   - 'V1', 'V1+', 'V1-', 'V1-2' (starts with V1, next char non-digit or end)
        #   - 'V0-1' (range notation where the specified grade is the upper bound)
        def _grade_pattern(g):
            n = re.match(r'^V(\d+)', g).group(1)
            return rf'(V{n}([^0-9]|$))|(V\d+-{n}([^0-9]|$))'
        pattern = '|'.join(_grade_pattern(g) for g in included_grades)
        plot_df = plot_df[plot_df['V-grade'].astype(str).str.contains(pattern, regex=True)]

    female_counts = (plot_df[plot_df[gender_column].isin(female_genders)]
                     .groupby('Route')['score'].count())
    male_counts = (plot_df[plot_df[gender_column].isin(male_genders)]
                   .groupby('Route')['score'].count())
    qualified_routes = (female_counts[female_counts >= min_counts].index
                        .intersection(male_counts[male_counts >= min_counts].index))
    print(f"{len(qualified_routes)} routes meet the minimum rating threshold "
          f"(>={min_counts} female, >={min_counts} male)")

    female_avg = (plot_df[plot_df[gender_column].isin(female_genders) &
                          plot_df['Route'].isin(qualified_routes)]
                  .groupby('Route')['score'].mean()
                  .rename('female_avg'))
    male_avg = (plot_df[plot_df[gender_column].isin(male_genders) &
                        plot_df['Route'].isin(qualified_routes)]
                .groupby('Route')['score'].mean()
                .rename('male_avg'))

    fm_df = pd.concat([female_avg, male_avg], axis=1).dropna()
    fm_df['diff'] = fm_df['female_avg'] - fm_df['male_avg']

    # Attach V-grade per route
    route_grade = plot_df.drop_duplicates('Route').set_index('Route')['V-grade']
    fm_df['V-grade'] = fm_df.index.map(route_grade)

    # --- Grade ordering: V0 < V1- < V1 < V1+ < V2- < V2 < V2+ < ... ---
    def grade_sort_key(g):
        m = re.match(r'V(\d+)([+-]?)', str(g))
        if not m:
            return -1
        n, mod = int(m.group(1)), m.group(2)
        return n * 3 + (0 if mod == '-' else 2 if mod == '+' else 1)

    present_grades = fm_df['V-grade'].dropna().unique()
    sorted_grades = sorted(present_grades, key=grade_sort_key)
    grade_colors = plt.cm.plasma(np.linspace(0, 0.85, len(sorted_grades)))
    grade_color_map = dict(zip(sorted_grades, grade_colors))

    n_routes = len(fm_df)
    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, n_routes * 0.35)))

    # --- Plot 1: Scatter with V-grade color gradient ---
    ax = axes[0]
    for grade in sorted_grades:
        subset = fm_df[fm_df['V-grade'] == grade]
        ax.scatter(subset['male_avg'], subset['female_avg'],
                   color=grade_color_map[grade], label=grade, zorder=3, s=60)
    no_grade = fm_df[fm_df['V-grade'].isna()]
    if not no_grade.empty:
        ax.scatter(no_grade['male_avg'], no_grade['female_avg'],
                   color='gray', label='unknown', zorder=3, s=60)
    lims = [
        min(fm_df['male_avg'].min(), fm_df['female_avg'].min()) - 0.1,
        max(fm_df['male_avg'].max(), fm_df['female_avg'].max()) + 0.1,
    ]
    ax.plot(lims, lims, 'k--', linewidth=1, label='y = x (no difference)')
    for route, row in fm_df.iterrows():
        ax.annotate(route, (row['male_avg'], row['female_avg']),
                    fontsize=7, textcoords='offset points', xytext=(4, 2))
    ax.set_xlabel('Male / mostly_male avg star rating')
    ax.set_ylabel('Female / mostly_female avg star rating')
    ax.set_title('Female vs Male Star Rating per Route')
    ax.legend(title='V-grade', fontsize=7, title_fontsize=8, loc='upper left')

    # --- Plot 2: Difference bar chart sorted by diff (female-preferred at top) ---
    fm_sorted = fm_df.sort_values('diff', ascending=True)
    ax = axes[1]
    colors = ['steelblue' if d < 0 else 'salmon' for d in fm_sorted['diff']]
    ax.barh(fm_sorted.index, fm_sorted['diff'], color=colors)
    ax.axvline(0, color='black', linewidth=0.8)

    x_pad = max(fm_sorted['diff'].abs().max() * 0.05, 0.05)
    for route, row in fm_sorted.iterrows():
        if row['diff'] >= 0:
            ax.text(row['diff'] + x_pad, route,
                    f"F:{row['female_avg']:.2f}  M:{row['male_avg']:.2f}",
                    va='center', ha='left', fontsize=6)
        else:
            ax.text(row['diff'] - x_pad, route,
                    f"F:{row['female_avg']:.2f}  M:{row['male_avg']:.2f}",
                    va='center', ha='right', fontsize=6)

    x_min, x_max = fm_sorted['diff'].min(), fm_sorted['diff'].max()
    ax.set_xlim(x_min - abs(x_min) * 0.5 - 0.3, x_max + abs(x_max) * 0.5 + 0.3)
    ax.set_xlabel('Difference (female avg − male avg)')
    ax.set_title('Rating Difference: Female/Mostly-Female\nvs Male/Mostly-Male (per route)')
    ax.tick_params(axis='y', labelsize=8)

    plt.tight_layout()
    plt.show()

    print(fm_df[['female_avg', 'male_avg', 'diff']].round(2))
