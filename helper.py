
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def fetch_medal_tally(df, year, country):
    df.rename(columns={'Year': 'year'}, inplace=True)

    flag = 0
    medal_df = df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'year', 'City', 'Sport', 'Event', 'Medal'])
    # Handling the different combinations of year and country
    if year == 'Overall' and country == 'Overall':
        temp_df = medal_df
    if year == 'Overall' and country != 'Overall':
        flag = 1
        temp_df = medal_df[medal_df['region'] == country]
    if year != 'Overall' and country == 'Overall':
        temp_df = medal_df[medal_df['year'] == int(year)]
    if year != 'Overall' and country != 'Overall':
        temp_df = medal_df[(medal_df['year'] == int(year)) & (medal_df['region'] == country)]

    if temp_df.empty:
        print("No data available for the given filters.")
        return
    if flag == 1:
        x = temp_df.groupby('year').sum()[['Gold', 'Silver', 'Bronze']].sort_values('year').reset_index()
    else:
        x = temp_df.groupby('region').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Gold', ascending=False).reset_index()
    x['total'] = x['Gold'] + x['Silver'] + x['Bronze']
    x['Gold'] = x['Gold'].astype('int')
    x['Silver'] = x['Silver'].astype('int')
    x['Bronze'] = x['Bronze'].astype('int')
    x['total'] = x['total'].astype('int')


    df.rename(columns={'year': 'Year'}, inplace=True)
    # medal_df.rename(columns={'year': 'Year'}, inplace=True)
    return x


def medal_tally(df):
    medal_tally = df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'])
    medal_tally = medal_tally.groupby('region').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Gold',
                                                                                                ascending=False).reset_index()
    medal_tally['total'] = medal_tally['Gold'] + medal_tally['Silver'] + medal_tally['Bronze']
    medal_tally['Gold']=medal_tally['Gold'].astype(int)
    medal_tally['Silver'] = medal_tally['Silver'].astype(int)
    medal_tally['Bronze'] = medal_tally['Bronze'].astype(int)

    return medal_tally
def country_year_list(df):
    years = df['Year'].unique().tolist()
    years.sort()
    years.insert(0, 'Overall')

    country = np.unique(df['region'].dropna().values).tolist()
    country.sort()
    country.insert(0, 'Overall')
    return years,country

def data_over_time(df,col):

    nations_over_time = df.drop_duplicates(['Year',col])['Year'].value_counts().reset_index().sort_values('Year')
    nations_over_time.rename(columns={'Year':'Edition','count':col},inplace=True)
    return nations_over_time


def most_successful(df, sport):
    # Filter rows with non-null medals
    temp_df = df.dropna(subset=['Medal'])

    # Filter by sport if specified
    if sport != 'Overall':
        temp_df = temp_df[temp_df['Sport'] == sport]

    # Count medals for each athlete
    athlete_medals = temp_df['Name'].value_counts().reset_index().head(10)
    athlete_medals.columns = ['Name', 'Medal_Count']

    # Merge to get additional details
    merged_df = athlete_medals.merge(df, left_on='Name', right_on='Name', how='left')[
        ['Name', 'Medal_Count', 'Sport', 'region']]

    # Drop duplicates and sort by Medal_Count
    x = merged_df.drop_duplicates(subset=['Name']).sort_values(by='Medal_Count', ascending=False)
    return x
def yearwise_medal_tally(df,country):
    temp_df=df.dropna(subset=['Medal'])
    temp_df.drop_duplicates(subset=['Team','NOC','Games','Year','City','Sport','Event','Medal'],inplace=True)
    new_df=temp_df[temp_df['region']==country]
    final_df=new_df.groupby('Year').count()['Medal'].reset_index()
    return final_df

def country_event_heatmap(df,country):
    temp_df = df.dropna(subset=['Medal'])
    temp_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'], inplace=True)
    new_df = temp_df[temp_df['region'] == country]
    pt = new_df.pivot_table(index='Sport', columns='Year', values='Medal', aggfunc='count').fillna(0)
    return pt


def most_successful_countrywise(df, country):
    temp_df = df.dropna(subset=['Medal'])
    temp_df = temp_df[temp_df['region'] == country]
    country_medals = temp_df['Name'].value_counts().reset_index().head(10)
    country_medals.columns = ['Name', 'Medal_Count']
    merged_df = country_medals.merge(df, left_on='Name', right_on='Name', how='left')[
        ['Name', 'Medal_Count', 'Sport', ]]
    x = merged_df.drop_duplicates(subset=['Name']).sort_values(by='Medal_Count', ascending=False)
    return x

def weight_v_height(df,sport):
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])
    athlete_df['Medal'].fillna('No Medal', inplace=True)
    if sport != 'Overall':
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        return temp_df
    else:
        return athlete_df
def men_vs_women(df):
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    men = athlete_df[athlete_df['Sex'] == 'M'].groupby('Year').count()['Name'].reset_index()
    women = athlete_df[athlete_df['Sex'] == 'F'].groupby('Year').count()['Name'].reset_index()

    final = men.merge(women, on='Year', how='left')
    final.rename(columns={'Name_x': 'Male', 'Name_y': 'Female'}, inplace=True)

    final.fillna(0, inplace=True)

    return final
def prepare_regression_data(df, country):
    # Filter the data for the selected country
    temp_df = df[df['region'] == country]
    # Group by year and calculate the total medals
    medal_data = temp_df.groupby('Year').sum()[['Gold', 'Silver', 'Bronze']].reset_index()
    medal_data['Total'] = medal_data['Gold'] + medal_data['Silver'] + medal_data['Bronze']
    return medal_data[['Year', 'Total']]

def train_regression_model(data):
    # Split data into features (X) and target (y)
    X = data[['Year']]
    y = data['Total']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2
def prepare_regression_data(df, country):
    # Filter the data for the selected country
    temp_df = df[df['region'] == country]
    # Group by year and calculate the total medals
    medal_data = temp_df.groupby('Year').sum()[['Gold', 'Silver', 'Bronze']].reset_index()
    medal_data['Total'] = medal_data['Gold'] + medal_data['Silver'] + medal_data['Bronze']
    return medal_data[['Year', 'Total']]

def train_regression_model(data):
    # Split data into features (X) and target (y)
    X = data[['Year']]
    y = data['Total']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2

