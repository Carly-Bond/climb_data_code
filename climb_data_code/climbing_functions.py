# import packages
from bs4 import BeautifulSoup # https://beautiful-soup-4.readthedocs.io/en/latest/
import copy
from datetime import datetime
import gender_guesser.detector as gender
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import requests # https://requests.readthedocs.io/en/latest/
import time
import urllib.parse



HEADERS = {'User-Agent': 'Independent research project - contact: carlybond907@gmail.com'}




# Define functions


### Functions related to grabbing information about the climbing area and climbs therein. ###

def retrieve_overview_csv(URL):
  """
  This function is designed to work with the output of "Route Finder" on 
  Mountain Project. The URL corresponds to the Mountain project page that has
  an "Export CSV" button at the top. 

  Args:
      URL: This is the URL of the webpage. Type is string. 

  Returns:
      return_type: pandas dataframe
  """

  csv_export_request = requests.get(URL, headers=HEADERS)

  if csv_export_request.status_code != 200:
    print("Something went wrong. Status code is: " + str(csv_export_request.status_code))
    return pd.DataFrame()

  if not csv_export_request.text.strip().startswith('Route'):
    print("Response does not look like a CSV. Mountain Project may have returned an HTML page.")
    print("First 500 chars of response:")
    print(csv_export_request.text[:500])
    return pd.DataFrame()

  csv_df = pd.read_csv(io.StringIO(csv_export_request.text))
  print("CSV converted to pandas df")


  csv_df['YDS']                 = csv_df['Rating'].str.extract(r'(5\.\d+[+-]?)', expand=False)
  csv_df['V-grade']             = csv_df['Rating'].str.extract(r'(V\d+(?:[+-]\d*)?)', expand=False)
  csv_df['Route Danger Rating'] = csv_df['Rating'].str.extract(r'\b(PG-?13|[RX])\b', expand=False)

  return csv_df


def get_route_URL(URL):
  """
  This function takes the URL of a specific climb and converts it to the URL 
  for the climbing statistics page. 

  Args:
      URL: This is the URL to the Mountain Project page for this climb. The type
      is string. 
      

  Returns:
      return_type: This is the URL to the Mountain Project stats page for this 
      climb. The type is string.  
  """

  substring = "/route/"
  text_to_insert = "stats/"
  return URL.replace(substring, substring + text_to_insert)



def get_route_stats(URL, stat_type, params = {'per_page': '250','page': '1',}, retries = 3, backoff = 2):
  """
  This function takes the URL of a specific climb and converts it to the URL 
  used to request the specified table on the route stats page.  

  Args:
      URL: This is the URL to the Mountain Project page for this climb. The type
           is string. 
      stat_type: This is the stat type that is requested. 
      params: Dictionary of parameters for the requested data. Default setting 
              is the default setting from MountainProject.
      

  Returns:
      return_type: This is a pandas dataframe of the requested route stat info.
      The type is pandas dataframe.  
  """
  prefix = "https://www.mountainproject.com/api/v2/routes/"
  
  if stat_type in ['stars', 'ratings', 'ticks', 'todos']:
    req_URL = prefix + URL.split('/')[-2] + "/" + stat_type
    wait = backoff
    for attempt in range(retries):
      try:
        response = requests.get(req_URL, params=params, headers=HEADERS)
        if response.status_code == 200 and response.text.strip():
          return pd.json_normalize(response.json()['data'])
        print(f"Attempt {attempt + 1} failed ({response.status_code}) for {req_URL}, retrying in {wait}s...")
      except requests.exceptions.RequestException as e:
        print(f"Attempt {attempt + 1} failed ({type(e).__name__}) for {req_URL}, retrying in {wait}s...")
      time.sleep(wait)
      wait *= 2
    print(f"All {retries} attempts failed for {req_URL}. Returning empty DataFrame.")
    return pd.DataFrame()

  else:
    print("This is not an accepted stat_type request.\
          \nTry 'stars', 'ratings', 'ticks', or 'todos'.")
    return pd.DataFrame()
  


def create_route_stats_df(route_name, star_df, rating_df, tick_df, prefix_col = ['id', 'date', 'createdAt', 'updatedAt']):
  """
  This function is designed to work with function fill_area_df_with_stats. It takes as input
  the individual stats dataframes for the specificed route, adds prefix's to unique columns that 
  share names across dataframes and performs a merge to create a dataframe that includes all of
  the stats. 


  Args:
      route_name: This is the name of the route, as listed on Mountain Project. Type is string.
      star_df: This is the dataframe of the star stats for this route. Type is pandas dataframe. 
      rating_df: This is the dataframe of the difficulty ratings for this route. Type is pandas dataframe.
      tick_df: This is the dataframe of user ticks and comments for this route. Type is pandas dataframe.
      prefix_col:   This is the list of columns that need a prefix indicating what stat dataframe this column
                    is affiliated with before the stat dataframes are merged together.


  Returns:
      return_type:  pandas dataframe. This is the combined stats for an individual climbing
                    route. route_stas_df.
  """ 
  # Prepare star_df for merge
  star_df.columns = [f'star_{i}' if i in prefix_col else f'{i}' for i in star_df.columns]

  # Prepare rating_df for merge
  cols_to_drop = rating_df.columns[(rating_df == 0).all()]
  rating_df.drop(columns=cols_to_drop, inplace = True)
  rating_df.columns = [f'rating_{i}' if i in prefix_col else f'{i}' for i in rating_df.columns]

  # Prepare tick_df for merge
  if 'user.id' in tick_df.columns:
    tick_df.dropna(subset=['user.id'], inplace = True)
    tick_df['user.id'] = tick_df['user.id'].astype('Int64')
  if 'comment' in tick_df.columns:
    #tick_df['comment'] = tick_df['text'].str.split(".", n = 1, expand = True).iloc[:, 1:]
    tick_df['comment'] = tick_df['text'].astype(str).str.split(".", n = 1, expand = True).get(1)
  cols_to_drop = ['user', 'text']
  for item in cols_to_drop[:]:
    if item not in tick_df.columns:
      cols_to_drop.remove(item)
  if cols_to_drop != []:
    tick_df.drop(columns=cols_to_drop, inplace = True)
  tick_df.columns = [f'tick_{i}' if i in prefix_col else f'{i}' for i in tick_df.columns]


  # merge with csv_df (aka overall route df)
  # Ensure empty DataFrames have at least the shared join columns so merge doesn't fail
  for df in [star_df, rating_df, tick_df]:
    for col in ['user.id', 'user.name']:
      if col not in df.columns:
        df[col] = pd.Series(dtype='object')
  route_stats_df = star_df.merge(rating_df, how = 'outer').merge(tick_df, how = 'outer')
  route_stats_df['user.id'] = route_stats_df['user.id'].astype('Int64')
  route_stats_df['Route'] = route_name

  return route_stats_df


def fill_area_df_with_stats(area_df, cache_path='route_stats_cache.pkl', delay=0.5, retries=3, backoff=2):
  """
  This function takes the overview pandas dataframe of a climbing area (the output of retrieve_overview_csv)
  and performs a request for each climbing route to retrieve the stats regarding stars, ratings, and ticks 
  for that route using the get_route_stats function. Once stats for all routes have been retrieved, they 
  are merged with the area overview dataframe and this resulting dataframe is returned. 

  Results are cached to disk after each route so that progress is preserved if the function is interrupted.
  On re-run, already-fetched routes are skipped.

  Args:
      area_df:    Type is pandas dataframe. Overview of the climbing area as downloaded from Mountain project
                  using function retrieve_overview_csv. 
      cache_path: Path to the pickle file used to cache results between runs. Default is 'route_stats_cache.pkl'.
      delay:      Seconds to wait between API calls. Default is 0.5.
      retries:    Number of retry attempts in get_route_stats on failure. Default is 3.
      backoff:    Initial wait time in seconds between retries (doubles each attempt). Default is 2.

  Returns:
      return_type:  Pandas dataframe. This is the original input dataframe, now merged
                    with stats(stars, ratings, and ticks) of each route.  
  """
  route_dict = {}
  for i in range(0, len(area_df)):
      route_dict.update({area_df.iat[i,0]: get_route_URL(area_df.iat[i,2])})
  print(route_dict)

  # Load from cache if available, otherwise start fresh
  if os.path.exists(cache_path):
      all_routes_stats_df = pd.read_pickle(cache_path)
      completed_routes = set(all_routes_stats_df['Route'].unique())
      print(f"Resuming — {len(completed_routes)} routes already cached.")
  else:
      all_routes_stats_df = pd.DataFrame(columns = ['star_id', 'score', 'star_createdAt', 'star_updatedAt', 'user.id',
                                                    'user.name', 'rating_id', 'allRatings', 'boulderRating', 'safteyRating',
                                                    'rating_createdAt', 'rating_updatedAt', 'tick_id', 'tick_date',
                                                    'comment', 'style', 'leadStyle', 'pitches', 'tick_createdAt',
                                                    'tick_updatedAt', 'Route']).astype({'user.id': 'Int64'})
      completed_routes = set()

  for keys in route_dict.keys():
      if keys in completed_routes:
          print(f"Skipping '{keys}' (already cached)")
          continue
      print('starting on ' + keys)
      star_df = get_route_stats(URL = route_dict[keys], stat_type = "stars", params = {'per_page': '250','page': '1'}, retries=retries, backoff=backoff)
      print('made star_df')
      time.sleep(delay)
      rating_df = get_route_stats(URL = route_dict[keys], stat_type = 'ratings', params = {'per_page': '250','page': '1'}, retries=retries, backoff=backoff)
      print('made rating_df')
      time.sleep(delay)
      tick_df = get_route_stats(URL = route_dict[keys], stat_type = 'ticks', params = {'per_page': '250','page': '1'}, retries=retries, backoff=backoff)
      print('made tick_df')
      route_stats_df = create_route_stats_df(route_name = keys, 
                                             star_df = star_df, 
                                             rating_df = rating_df, 
                                             tick_df= tick_df, 
                                             prefix_col = ['id', 'date', 'createdAt', 'updatedAt'])
      print("made route_stats_df")
      all_routes_stats_df = pd.concat([all_routes_stats_df, route_stats_df])
      all_routes_stats_df.to_pickle(cache_path)
      print("Finished updating all_routes_stats_df with stats from " + route_dict[keys])
  return all_routes_stats_df
 
    
 #### Functions related to capturing user information #### 


def user_apply_func(x):
    '''
    This functions is designed to apply row-by-row to a pandas dataframe. 
    It takes as input the dataframe of all relevant users (listing their user ID and username) and initiates the individual user dicitonary.

    Args:
        x: Pandas dataframe of relevant users, listing their user ID and username.

    Returns:
        user_dict: dictionary of an individual user, with values assigned to the keys 'user_ID' and 'user_name', and more keys that have no values at this step. 
    '''
    # initiate dictionary
    user_dict = dict.fromkeys(['user_ID', 'user_name', 'user_URL', 'request_date', 'location', 'age_at_request_date', 'listed_gender', 'guessed_gender'])
    # fill the dictionary values
    user_dict['user_ID'] = x['user.id']
    user_dict['user_name'] = x['user.name']
    return user_dict



def get_user_info(user_dict, retries=3, backoff=2):
  """
  This function takes the dictionary of an individual user, containing their
  user ID and name. It returns the dictionary of an individual user updated with
  the age of the user at time of request, the date of data request, the 
  user's gender, and the user's location of residence. 


  Args:
      user_dict: Dictionary of individual user
      retries:    Number of retry attempts in get_route_stats on failure. Default is 3.
      backoff:    Initial wait time in seconds between retries (doubles each attempt). Default is 2.
      
      

  Returns:
      return_type: a dictionary of user information. 
      {user_ID, user_name, listed_gender, guessed_gender, age_at_request_date, 
       request_date, location}
  """

  user_dict['user_URL'] = "https://www.mountainproject.com/user/" + str(user_dict['user_ID']) + "/" + urllib.parse.quote(user_dict['user_name'].lower().replace(" ", "-"))


  # page = requests.get(user_dict['user_URL']) # Step 1: get the page content
  # soup = BeautifulSoup(page.text, 'html.parser')
  wait = backoff
  for attempt in range(retries):
      try:
          page = requests.get(user_dict['user_URL'], headers=HEADERS)
          break
      except requests.exceptions.ConnectionError as e:
          print(f"Attempt {attempt + 1} failed for{user_dict['user_URL']}: {e}")
          if attempt + 1 == retries:
              print("All retries failed, skipping user.")
              return user_dict
          time.sleep(wait)
          wait *= 2
  soup = BeautifulSoup(page.text, 'html.parser')



  lines = soup.find_all("div", class_ = "")

  line_list = []
  for line in lines:
    if line.get_text().strip() != '':
      line_list.append(line.get_text().strip())

  #print(line_list)


  if len(line_list) > 0:
    line_list = line_list[0].replace("  ", "").split('\n')
    print(user_dict['user_name'] + ": line_list exists")
  else:
    print(user_dict['user_name'] + ": line_list is empty")

  #print(line_list)


  for i in range(0, len(line_list)):
    if len(line_list[i]) < 3:
      continue
    elif "years old" in line_list[i]:
      user_dict['age_at_request_date'] = int(line_list[i].split()[0])
    elif line_list[i] in ('Male', 'Female'):
      user_dict['listed_gender'] = line_list[i]
    else:
      user_dict['location'] = line_list[i]

    user_dict['request_date'] = datetime.now().year


    
  user_dict["guessed_gender"] = gender.Detector().get_gender(user_dict['user_name'].split(" ")[0].title())




  return user_dict


def fill_user_dict_with_info(all_user_dict, cache_path='user_info_cache.pkl', delay=2):
      """
      This is a wrapper function that calls get_user_info(). The wrapper makes sure that the cache is checked and adds a delay between requests. 

      Args:
        all_user_dict: Dictionary of all relevant users
        cache_path: Cache to saved path of users that have already been processed
        delay:  Seconds to wait between API calls. Default is 2 because user profiles have more stringent reactions to scraping.


      Returns:
        all_user_dict: Dictionary of relevant users, with updated information from Mountain Project profiles. 

      """
      # Load cache if available
      if os.path.exists(cache_path):
          with open(cache_path, 'rb') as f:
              cached_dict = pickle.load(f)
          completed_ids = set(cached_dict.keys())
          # Merge cached entries back into all_user_dict
          for k, v in cached_dict.items():
              if k in all_user_dict:
                  all_user_dict[k] = v
          print(f"Resuming — {len(completed_ids)} users already cached.")
      else:
          completed_ids = set()

      for key in all_user_dict.keys():
          if key in completed_ids:
              print(f"Skipping user {key} (already cached)")
              continue
          get_user_info(user_dict=all_user_dict[key])
          time.sleep(delay)
          # Save after each user
          with open(cache_path, 'wb') as f:
              pickle.dump(all_user_dict, f)

      return all_user_dict



def remove_failed_users_from_cache(cache_path='user_info_cache.pkl', dry_run=False):
    """
    Removes entries from the user info cache where data retrieval failed
    (i.e., request_date is None). This allows fill_user_dict_with_info to
    retry those users on the next run.

    Args:
        cache_path: Path to the user info cache pickle file. Default is 'user_info_cache.pkl'.
        dry_run:    If True, print what would be removed without modifying the cache. Default is False.

    Returns:
        removed_ids: List of user IDs that were (or would be) removed.
    """
    if not os.path.exists(cache_path):
        print(f"Cache file '{cache_path}' not found.")
        return []

    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    failed_ids = [uid for uid, info in cache.items() if info.get('request_date') is None]

    if not failed_ids:
        print("No failed entries found in cache.")
        return []

    print(f"Found {len(failed_ids)} failed entries (request_date is None).")

    if dry_run:
        print("Dry run — no changes made. Failed user IDs:")
        for uid in failed_ids:
            print(f"  {uid}: {cache[uid].get('user_name')}")
        return failed_ids

    for uid in failed_ids:
        del cache[uid]

    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)

    print(f"Removed {len(failed_ids)} failed entries from cache. They will be retried on next run.")
    return failed_ids


def fill_missing_guessed_gender(user_info_df):
    """
    For any row in user_info_df where guessed_gender is NA, infers the gender
    from the first word of user_name using gender_guesser and fills it in.

    Args:
        user_info_df: DataFrame with at least 'user_name' and 'guessed_gender' columns.

    Returns:
        user_info_df with guessed_gender filled in where it was NA.
    """
    detector = gender.Detector()
    mask = user_info_df['guessed_gender'].isna()
    user_info_df.loc[mask, 'guessed_gender'] = user_info_df.loc[mask, 'user_name'].apply(
        lambda name: detector.get_gender(str(name).split(" ")[0].title())
    )
    print(f"Filled guessed_gender for {mask.sum()} rows.")
    return user_info_df


def assign_likely_gender(row):
      if pd.notna(row['listed_gender']):
          return row['listed_gender'].lower()
      return str(row['guessed_gender']).lower() if pd.notna(row['guessed_gender']) else None




def make_all_user_dict(area_df, area_name):
    """
    This function takes the area with stats dataframe and returns a dataframe of all of the unique users and a dictionary version of that dataframe.
    """
    unique_user_df = (
        area_df[['user.id', 'user.name']]
        .dropna()
        .drop_duplicates()
        .copy()
    )

    unique_user_df['user.id'] = unique_user_df['user.id'].astype(int)

    occurrence_counts = area_df['user.id'].dropna().astype(int).value_counts()
    unique_user_df[f"{area_name} occurences"] = unique_user_df['user.id'].map(occurrence_counts)
    unique_user_df = unique_user_df.sort_values(f"{area_name} occurences", ascending=False)

    unique_user_df['user_dict'] = unique_user_df.apply(user_apply_func, axis=1)
    unique_user_df.set_index('user.id', inplace=True)
    all_user_dict = unique_user_df['user_dict'].to_dict()

    print(f"Unique users found: {len(all_user_dict)}")

    return unique_user_df, all_user_dict 