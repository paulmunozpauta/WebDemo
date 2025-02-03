import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI environments
from flask import Flask, render_template, request, url_for, session, send_file, make_response, redirect, flash
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import seaborn as sns
import os
import logging
import requests
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
from flask_mail import Mail, Message
import io  # Import io module
import csv  # Make sure csv is also imported if not already
import secrets
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from itertools import chain
from werkzeug.security import generate_password_hash, check_password_hash
from flask_migrate import Migrate

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Secure secret key

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.yourmailserver.com'  #not configured yet
app.config['MAIL_PORT'] = 587  #not configured yet
app.config['MAIL_USERNAME'] = 'your-email@example.com' #not configured yet
app.config['MAIL_PASSWORD'] = 'your-email-password' #not configured yet
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

mail = Mail(app)

# Determine the path of the database file relative to the script location
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

db_path = os.path.join(basedir, 'impactdb.v1.2.dg_filled.db')
users_db_path = os.path.join(basedir, 'users.db')



# Ensure the database file exists
if not os.path.exists(db_path):
    logging.error(f"Database file {db_path} does not exist.")
    raise FileNotFoundError(f"Database file {db_path} does not exist.")

# Configure the app to use multiple databases



# Configure SQLAlchemy with multiple databases
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_BINDS'] = {
    'users': f'sqlite:///{users_db_path}'
}
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False




db = SQLAlchemy(app)
migrate = Migrate(app, db)


# Example model for migration
class ExampleModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)

class User(db.Model):
    __bind_key__ = 'users'  # Specify the database bind
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    lastname = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=False)
    institution = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(255), nullable=False)  # Hashed password



@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        lastname = request.form['lastname']
        username = request.form['username']
        institution = request.form['institution']
        email = request.form['email']
        password = request.form['password']

        # Check for unique username
        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose a different username.', 'danger')
            return redirect(url_for('register'))

        # Hash the password
        hashed_password = generate_password_hash(password)

        # Create a new user
        user = User(
            name=name,
            lastname=lastname,
            username=username,
            institution=institution,
            email=email,
            password=hashed_password
        )
        db.session.add(user)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Check if the user exists in the database
        user = User.query.filter_by(email=email).first()

        if not user:
            # If email is not found, flash an error message
            flash('Email not found. Please register or try again.', 'danger')
            return redirect(url_for('login'))

        if not check_password_hash(user.password, password):
            # If the password is incorrect, flash an error message
            flash('Incorrect password. Please try again.', 'danger')
            return redirect(url_for('login'))

        # If both email and password are correct, log the user in
        session['user_id'] = user.id
        session['username'] = user.username
        flash('Login successful!', 'success')
        return redirect(url_for('home'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))


class TotalSummary(db.Model):
    __tablename__ = 'Total_Summary'
    
    Event_ID = db.Column(db.Text, primary_key=True)
    Event_Names = db.Column(db.Text)
    Sources = db.Column(db.Text)  
    Main_Event = db.Column(db.Text)
    Hazards = db.Column(db.Text)  
    Administrative_Areas_Norm = db.Column(db.Text)
    Administrative_Areas_GID = db.Column(db.Text)
    Administrative_Areas_Type = db.Column(db.Text)
    Administrative_Areas_GeoJson = db.Column(db.Text)
    Start_Date_Day = db.Column(db.Integer)
    Start_Date_Month = db.Column(db.Integer)
    Start_Date_Year = db.Column(db.Integer)
    End_Date_Day = db.Column(db.Integer)
    End_Date_Month = db.Column(db.Integer)
    End_Date_Year = db.Column(db.Integer)
    
    Total_Deaths_Min = db.Column(db.Float)
    Total_Deaths_Max = db.Column(db.Float)
    Total_Deaths_Approx = db.Column(db.Integer)
    
    Total_Injuries_Min = db.Column(db.Float)
    Total_Injuries_Max = db.Column(db.Float)
    Total_Injuries_Approx = db.Column(db.Integer)
    
    Total_Affected_Min = db.Column(db.Float)
    Total_Affected_Max = db.Column(db.Float)
    Total_Affected_Approx = db.Column(db.Integer)
    
    Total_Displaced_Min = db.Column(db.Float)
    Total_Displaced_Max = db.Column(db.Float)
    Total_Displaced_Approx = db.Column(db.Float)
    
    Total_Homeless_Min = db.Column(db.Float)
    Total_Homeless_Max = db.Column(db.Float)
    Total_Homeless_Approx = db.Column(db.Integer)
    
    Total_Buildings_Damaged_Min = db.Column(db.Float)
    Total_Buildings_Damaged_Max = db.Column(db.Float)
    Total_Buildings_Damaged_Approx = db.Column(db.Float)
    
    Total_Insured_Damage_Min = db.Column(db.Float)
    Total_Insured_Damage_Max = db.Column(db.Float)
    Total_Insured_Damage_Approx = db.Column(db.Float)
    Total_Insured_Damage_Unit = db.Column(db.Text)
    Total_Insured_Damage_Inflation_Adjusted = db.Column(db.Float)
    Total_Insured_Damage_Inflation_Adjusted_Year = db.Column(db.Integer)
    
    Total_Damage_Min = db.Column(db.Float)
    Total_Damage_Max = db.Column(db.Float)
    Total_Damage_Approx = db.Column(db.Float)
    Total_Damage_Unit = db.Column(db.Text)
    Total_Damage_Inflation_Adjusted = db.Column(db.Float)
    Total_Damage_Inflation_Adjusted_Year = db.Column(db.Integer)





@app.route('/')
def home():
    return render_template('index.html')




@app.route('/search', methods=['GET', 'POST'])
def search():
    # Retrieve unique countries by splitting lists stored in the database
    countries_query = db.session.query(TotalSummary.Administrative_Areas_Norm).distinct().all()
    countries = set(chain.from_iterable(
        eval(country[0]) if country[0].startswith('[') else [country[0]] for country in countries_query
    ))
    countries = sorted(countries)

    events = db.session.query(TotalSummary.Main_Event).distinct().all()
    events = [event[0] for event in events]

    if request.method == 'POST':
        # Collect filter inputs
        country = request.form.get('country')
        event_type = request.form.get('event_type')
        impact = request.form.get('impact')
        impact_value = request.form.get('impact_value')
        since_year = request.form.get('since_year')

        # Build query based on filters
        query = TotalSummary.query
        if country:
            query = query.filter(TotalSummary.Administrative_Areas_Norm.contains(country))
        if event_type and event_type != 'all':
            query = query.filter(TotalSummary.Main_Event == event_type)
        if impact and impact_value:
            query = query.filter(getattr(TotalSummary, impact) >= float(impact_value))
        if since_year:
            query = query.filter(TotalSummary.Start_Date_Year >= int(since_year))

        # Execute query and fetch results
        results = query.all()

        # Convert results to dictionary format for rendering
        results_dict = [
            {
                'Event_Names': ', '.join(result.Event_Names) if isinstance(result.Event_Names, list) else result.Event_Names,
                'Main_Event': result.Main_Event,
                'Country': result.Administrative_Areas_Norm,
                'Start_Date': f"{result.Start_Date_Year}-{result.Start_Date_Month}-{result.Start_Date_Day}",
                'End_Date': f"{result.End_Date_Year}-{result.End_Date_Month}-{result.End_Date_Day}",
                'Total_Deaths': result.Total_Deaths_Max if result.Total_Deaths_Max is not None else 'N/A',
                'Total_Damage': result.Total_Damage_Max if result.Total_Damage_Max is not None else 'N/A'
            }
            for result in results
        ]

        # Store results in the session for CSV download
        session['search_results'] = results_dict

        return render_template('search_results.html', results=results_dict)

    return render_template('search.html', countries=countries, events=events)






@app.route('/visualization')
def visualization():
    viz_type = request.args.get('viz_type', 'distributions')
    sub_type = request.args.get('sub_type', 'event_type')
    image_path = None  # Initialize image_path to avoid the "referenced before assignment" error

    # Define the custom colors
    custom_colors = {
        'Flood': '#1f77b4',  # blue
        'Drought': '#ff7f0e',  # orange
        'Wildfire': '#d62728',  # red
        'Tornado': '#9467bd',  # purple
        'Extratropical Storm/Cyclone': '#2ca02c',  # green
        'Tropical Storm/Cyclone': '#bcbd22',  # yellow-green
        'Extreme Temperature': '#e377c2',  # pink
    }


    try:
        logging.debug(f"Database path: {db_path}")
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM Total_Summary", conn)
        conn.close()

        # Handle missing date components using pd.NaT
        def construct_start_date(row):
            if pd.isna(row['Start_Date_Year']):
                return pd.NaT
            start_month = row['Start_Date_Month'] if not pd.isna(row['Start_Date_Month']) else pd.NaT
            start_day = row['Start_Date_Day'] if not pd.isna(row['Start_Date_Day']) else pd.NaT
            try:
                return pd.Timestamp(year=int(row['Start_Date_Year']),
                                    month=int(start_month) if not pd.isna(start_month) else 1,
                                    day=int(start_day) if not pd.isna(start_day) else 1)
            except ValueError:
                return pd.NaT

        def construct_end_date(row):
            if pd.isna(row['End_Date_Year']):
                return pd.NaT
            end_month = row['End_Date_Month'] if not pd.isna(row['End_Date_Month']) else 1
            end_day = row['End_Date_Day'] if not pd.isna(row['End_Date_Day']) else 1
            try:
                return pd.Timestamp(year=int(row['End_Date_Year']),
                                    month=int(end_month),
                                    day=int(end_day))
            except ValueError:
                return pd.NaT

        # Construct Start_Date and End_Date
        df['Start_Date'] = df.apply(construct_start_date, axis=1)
        df['End_Date'] = df.apply(construct_end_date, axis=1)
        df['Start_Date'].fillna(df['End_Date'], inplace=True)

        # Handle country normalization
        df['Country_Norm'] = df['Administrative_Areas_Norm'].str.strip("[]").str.replace("'", "").str.split(", ")
        flat_country_data = df.explode('Country_Norm')

        sns.set(style="whitegrid")  # Set Seaborn style globally

        # Check for distributions visualization type
        if viz_type == 'distributions':
            if sub_type == 'event_type':
                # Countplot for event types
                plt.figure(figsize=(10, 6))
                sns.countplot(y='Main_Event', data=df, order=df['Main_Event'].value_counts().index)
                plt.title('Number of historic events by type')
                plt.xlabel('Count')
                plt.ylabel('Event Type')
                image_path = 'event_type.svg'


            elif sub_type == 'event_type_pie':
                # Generate the pie chart
                event_counts = df['Main_Event'].value_counts()
                colors = [custom_colors.get(event, '#333333') for event in event_counts.index]

                plt.figure(figsize=(4, 4))
                plt.pie(
                    event_counts,
                    labels=None,
                    colors=colors,
                    autopct='%1.1f%%',
                    startangle=90
                )
                plt.axis('equal')
                plt.title('Distribution of Events by Type')
                image_path = 'event_type_pie_chart.svg'
            
            elif sub_type == 'decadal_event_distribution':
            
                # Add the decadal grouped bar chart
                df['Start_Date_Year'] = pd.to_numeric(df['Start_Date_Year'], errors='coerce')
                df = df.dropna(subset=['Start_Date_Year'])
                df['Decade_Class'] = (df['Start_Date_Year'] // 10) * 10
                event_decade_counts = df.groupby(['Decade_Class', 'Main_Event']).size().reset_index(name='Count')
                event_decade_counts_pivot = event_decade_counts.pivot(index='Decade_Class', columns='Main_Event', values='Count').fillna(0)

                fig, ax = plt.subplots(figsize=(10, 6))
                event_decade_counts_pivot.plot(kind='bar', stacked=True, ax=ax, color=[custom_colors.get(event, '#333333') for event in event_decade_counts_pivot.columns])
                #plt.title('Decadal distribution of main events')
                plt.xlabel('Decade')
                plt.ylabel('Number of events')
                plt.xticks(rotation=45)
                plt.legend()
                image_path = 'decadal_event_distribution.svg'

            elif sub_type == 'impact_bubble_chart':
                
                # Columns related to impacts
                impact_columns = [
                    'Total_Deaths_Max', 'Total_Injuries_Max', 'Total_Affected_Max', 
                    'Total_Displaced_Max', 'Total_Homeless_Max', 'Total_Buildings_Damaged_Max',
                    'Total_Damage_Max', 'Total_Insured_Damage_Max'
                ]
                
                # Ensure impact columns are numeric
                for column in impact_columns:
                    df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0)
    
                # Prepare data for bubble chart
                bubble_data = []
                xtick_labels = {
                    'Total_Deaths_Max': 'Deaths',
                    'Total_Injuries_Max': 'Injuries',
                    'Total_Affected_Max': 'Affected',
                    'Total_Displaced_Max': 'Displaced',
                    'Total_Homeless_Max': 'Homeless',
                    'Total_Buildings_Damaged_Max': 'Buildings Damaged',
                    'Total_Damage_Max': 'Total Damage',
                    'Total_Insured_Damage_Max': 'Insured Damage'
                }
    
                # Collect bubble data
                for event_type in df['Main_Event'].unique():
                    event_data = df[df['Main_Event'] == event_type]
                    impact_sums = event_data[impact_columns].sum()
                    
                    for impact, value in impact_sums.items():
                        bubble_data.append({
                            'Impact': xtick_labels.get(impact, impact),  # Use custom x-tick labels
                            'Event': event_type,
                            'Impact_Sum': value,
                            'Color': custom_colors.get(event_type, '#333333'),  # Apply custom color for event
                            'Size': value  # Bubble size is proportional to the impact sum
                        })
    
                # Convert bubble data to DataFrame
                bubble_df = pd.DataFrame(bubble_data)
    
                # Use log of the impact sums to better visualize bubble size differences
                bubble_df['Log_Size'] = np.log10(bubble_df['Size'] + 1) * 400  # Add 1 to avoid log(0)
    
                # Create the bubble chart with better scaling
                plt.figure(figsize=(12, 12))
                sc = plt.scatter(
                    x=bubble_df['Impact'],
                    y=bubble_df['Event'],
                    s=bubble_df['Log_Size'],  # Bubble size
                    c=bubble_df['Color'],
                    alpha=0.6,
                    edgecolor='w',
                    linewidth=1
                )
    
                plt.xlabel('Impact Type', fontsize=14)
                plt.ylabel('Main Event', fontsize=14)
                plt.title('Bubble Chart of Total Max Impact by Main Event and Impact Type (Log Scale)', fontsize=16)
                plt.xticks(rotation=45, fontsize=14)
                plt.yticks(fontsize=14)
    
                # Adjust legend to show specific size examples
                legend_values = [1e3, 1e5, 1e7, 1e9, 1e11]  # Customize legend values
                legend_labels = [f'{int(value):,}' for value in legend_values]
    
                for value, label in zip(legend_values, legend_labels):
                    plt.scatter([], [], s=np.log10(value + 1) * 400, c='gray', alpha=0.6, edgecolor='w', label=label)
    
                plt.legend(title='Impact Size (Log Scale)', loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=5)
                
                image_path = 'impact_bubble_chart.svg'
            
            elif sub_type == 'event_countries':
                # Pie chart for distribution of events by country
                plt.figure(figsize=(10, 7))
                top_5_countries = flat_country_data['Country_Norm'].value_counts().nlargest(5)
                remaining_events = flat_country_data['Country_Norm'].value_counts().sum() - top_5_countries.sum()
                country_labels = list(top_5_countries.index) + ['Other countries']
                country_sizes = list(top_5_countries.values) + [remaining_events]
                plt.pie(country_sizes, labels=country_labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('husl', len(country_labels)))
                plt.axis('equal')
                plt.title('Distribution of Events by Country')
                image_path = 'event_countries.svg'

            elif sub_type == 'event_continent':
                # Pie chart for distribution of events by continent
                plt.figure(figsize=(10, 7))
                continent_mapping = {
                    'United States': 'North America',
                    'Russia': 'Europe',
                    'South Korea': 'Asia',
                    'North Korea': 'Asia',
                    'United Kingdom': 'Europe',
                    'France': 'Europe',
                    'Germany': 'Europe',
                    'Australia': 'Oceania',
                    'India': 'Asia',
                    # Add more country-to-continent mappings as needed
                }
                flat_country_data['Continent'] = flat_country_data['Country_Norm'].map(continent_mapping)
                continent_counts = flat_country_data['Continent'].value_counts()
                plt.pie(continent_counts.values, labels=continent_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('husl', len(continent_counts)))
                plt.axis('equal')
                plt.title('Distribution of Events by Continent')
                image_path = 'event_continent.svg'

            elif sub_type == 'event_trend':
                # Cumulative event trend plot
                plt.figure(figsize=(10, 6))
                df_resampled_all = df.set_index('Start_Date').resample('M').size().cumsum().reset_index()
                df_resampled_all['Event_Type'] = 'All Events'
                df_resampled_all = df_resampled_all.rename(columns={0: 'Event_Count'})
                df_resampled_types = df.groupby('Main_Event').apply(
                    lambda x: x.set_index('Start_Date').resample('M').size().cumsum()
                ).reset_index().rename(columns={0: 'Event_Count', 'Main_Event': 'Event_Type'})
                df_combined = pd.concat([df_resampled_all, df_resampled_types], ignore_index=True)
                df_total = df_combined[df_combined['Event_Type'] == 'All Events']
                df_events = df_combined[df_combined['Event_Type'] != 'All Events']
                palette = sns.color_palette("husl", df_events['Event_Type'].nunique())
                sns.lineplot(x='Start_Date', y='Event_Count', hue='Event_Type', data=df_events, palette=palette)
                plt.plot(df_total['Start_Date'], df_total['Event_Count'], color='black', linewidth=2.5, label='All Events')
                plt.xlim(pd.Timestamp('1800-01-01'), df_combined['Start_Date'].max())
                plt.title('Number of worldwide events over time')
                plt.xlabel('Date')
                plt.ylabel('Number of Events')
                plt.xticks(rotation=45)
                plt.legend(title='Event Type')
                image_path = 'event_trend.svg'

        # Check for map visualization type
        elif viz_type == 'map':
            naturalearth_path = 'wikimpacts_web/data/naturalearth/ne_110m_admin_0_countries.shp'
            world = gpd.read_file(naturalearth_path)

            if sub_type == 'event_location':
                # Flatten the country data and ensure it's string-based
                flat_country_data = flat_country_data.explode('Country_Norm')

                # Count the number of events per country
                event_counts_by_country = flat_country_data['Country_Norm'].value_counts().reset_index()
                event_counts_by_country.columns = ['Country', 'Number of Events']

                # Merge event counts with the world dataset
                world_events = world.merge(event_counts_by_country, left_on='ADMIN', right_on='Country', how='left')

                # Fill missing event counts with 0
                world_events['Number of Events'].fillna(0, inplace=True)

                # Plot the event locations on a map
                fig, ax = plt.subplots(1, 1, figsize=(15, 10))
                cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', 'orangered'], N=256)
                world_events.plot(column='Number of Events', cmap=cmap, linewidth=0.8, ax=ax, edgecolor='0.8', legend=False)

                # Add a color bar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=world_events['Number of Events'].min(), vmax=world_events['Number of Events'].max()))
                sm._A = []  # fake the array of the scalar mappable
                cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.036, pad=0.1)
                cbar.set_label('Number of Events')

                # Add axis labels
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')

                # Save the map as an image
                image_path = 'event_location.svg'

            elif sub_type == 'event_floods':
                
# Filter for flood events
                flood_events = df[df['Main_Event'] == 'Flood']

                # Flatten the country data for each flood event
                flood_events = flood_events.explode('Country_Norm')

                # Count the number of flood events per country
                flood_event_counts = flood_events['Country_Norm'].value_counts().reset_index()
                flood_event_counts.columns = ['Country', 'Number of Events']

                # Merge flood event counts with the world dataset
                world_flood_events = world.merge(flood_event_counts, left_on='ADMIN', right_on='Country', how='left')

                # Fill missing event counts with 0
                world_flood_events['Number of Events'].fillna(0, inplace=True)

                # Plot the flood events on a map
                fig, ax = plt.subplots(1, 1, figsize=(15, 10))
                cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', 'blue'], N=256)
                world_flood_events.plot(column='Number of Events', cmap=cmap, linewidth=0.8, ax=ax, edgecolor='0.8', legend=False)

                # Add a color bar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=world_flood_events['Number of Events'].min(), vmax=world_flood_events['Number of Events'].max()))
                sm._A = []  # fake the array of the scalar mappable
                cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.036, pad=0.1)
                cbar.set_label('Number of Flood Events')

                # Add axis labels
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')

                # Save the map as an image
                image_path = 'event_floods.svg'                
                

            elif sub_type == 'event_wildfires':
                # Filter for wildfire events
                wildfire_events = df[df['Main_Event'] == 'Wildfire']

                # Flatten the country data
                wildfire_events = wildfire_events.explode('Country_Norm')

                # Count the number of wildfire events per country
                wildfire_event_counts = wildfire_events['Country_Norm'].value_counts().reset_index()
                wildfire_event_counts.columns = ['Country', 'Number of Events']

                # Merge wildfire event counts with the world dataset
                world_wildfire_events = world.merge(wildfire_event_counts, left_on='ADMIN', right_on='Country', how='left')

                # Fill missing event counts with 0
                world_wildfire_events['Number of Events'].fillna(0, inplace=True)

                # Plot the wildfire events on a map
                fig, ax = plt.subplots(1, 1, figsize=(15, 10))
                cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', 'red'], N=256)
                world_wildfire_events.plot(column='Number of Events', cmap=cmap, linewidth=0.8, ax=ax, edgecolor='0.8', legend=False)

                # Add a color bar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=world_wildfire_events['Number of Events'].min(), vmax=world_wildfire_events['Number of Events'].max()))
                sm._A = []  # fake the array of the scalar mappable
                cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.036, pad=0.1)
                cbar.set_label('Number of Wildfire Events')

                # Add axis labels
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')

                # Save the map as an image
                image_path = 'event_wildfires.svg'                
        
                
        elif viz_type == 'impacts':
            impact_columns = [
                'Total_Deaths_Max', 'Total_Injuries_Max', 'Total_Affected_Max',
                'Total_Displaced_Max', 'Total_Homeless_Max', 'Total_Buildings_Damaged_Max',
                'Total_Damage_Max', 'Total_Insured_Damage_Max'
            ]
        
            # Define impact column labels and corresponding titles
            xtick_labels = {
                'Total_Deaths_Max': 'Deaths',
                'Total_Injuries_Max': 'Injuries',
                'Total_Affected_Max': 'Affected',
                'Total_Displaced_Max': 'Displaced',
                'Total_Homeless_Max': 'Homeless',
                'Total_Buildings_Damaged_Max': 'Buildings Damaged',
                'Total_Damage_Max': 'Total Damage',
                'Total_Insured_Damage_Max': 'Insured Damage'
            }
        
            title_mapping = {
                'Total_Deaths_Max': 'Number of deaths',
                'Total_Injuries_Max': 'Number of people injured',
                'Total_Affected_Max': 'Number of people affected',
                'Total_Displaced_Max': 'Number of displaced',
                'Total_Homeless_Max': 'Number of homeless',
                'Total_Buildings_Damaged_Max': 'Number of buildings damaged',
                'Total_Damage_Max': 'Total damage in 2024 USD',
                'Total_Insured_Damage_Max': 'Total insured damage in 2024 USD'
            }
        
            # Ensure numerical columns are treated correctly
            for column in impact_columns:
                df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0)
        
            # Define the specific order of event types
            event_order = [
                'Extratropical Storm/Cyclone', 'Tropical Storm/Cyclone', 'Tornado',
                'Flood', 'Drought', 'Wildfire', 'Extreme Temperature'
            ]
        
            if sub_type in impact_columns:
                # Aggregate data by event type and sort by the predefined order
                impact_data = df.groupby('Main_Event')[sub_type].sum()
                impact_data = impact_data.reindex(event_order)
        
                # Create the plot for the selected impact
                plt.figure(figsize=(8, 6))
                sns.barplot(
                    x=impact_data.index,
                    y=impact_data.values,
                    palette=[custom_colors.get(event, '#333333') for event in impact_data.index]
                )
                plt.yscale('log')  # Log scale for better representation
                plt.xticks(rotation=45, fontsize=10)
                plt.ylabel('Impact (Log Scale)', fontsize=10)
                plt.xlabel('')  # Remove x-label
                plt.title(f'{title_mapping.get(sub_type, sub_type)} by event', fontsize=12)
                plt.tight_layout()
        
                # Save the plot to the static directory
                static_dir = os.path.join(basedir, 'wikimpacts_web', 'static')
                if not os.path.exists(static_dir):
                    os.makedirs(static_dir)
        
                image_path = f'impact_metric_{sub_type}.svg'
                
               


        # Check if static directory exists, and save the plot
        static_dir = os.path.join(basedir, 'wikimpacts_web', 'static')
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)

        if image_path:
            plt.tight_layout()
            plt.savefig(os.path.join(static_dir, image_path), format='svg', dpi=600)
            plt.close()
            return render_template('visualization.html', image_path=image_path)
        else:
            return "Visualization not created. Please check the options."
    
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        return str(e)



@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/news')
def news():
    return render_template('news.html')
    
@app.route('/publications')
def publications():
    return render_template('publications.html')    
    
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        email = request.form['email']
        subject = request.form['subject']
        message = request.form['message']

        # Compose email
        msg = Message(subject, sender=email, recipients=['recipient-email@vub.be'])
        msg.body = message

        # Send email
        try:
            mail.send(msg)
            flash('Thank you for contacting us! Your message has been sent successfully.', 'success')
        except Exception as e:
            flash('Failed to send message. Please try again later.', 'danger')

        return redirect(url_for('contact'))

    return render_template('contact.html')

# Ensure that the route name matches the 'action' in your HTML form
@app.route('/send_email', methods=['POST'])
def send_email():
    # Here we handle the form submission logic
    email = request.form['email']
    subject = request.form['subject']
    message = request.form['message']

    # Send the email using Flask-Mail
    msg = Message(subject, sender=email, recipients=['paul.munoz@vub.be'])
    msg.body = message

    try:
        mail.send(msg)
        flash('Your message has been sent successfully!', 'success')
    except Exception as e:
        flash('An error occurred while sending your message. Please try again.', 'danger')

    return redirect(url_for('contact'))


@app.route('/download_csv')
def download_csv():
    # Get the search results from the session (now in JSON-serializable format)
    results = session.get('search_results', [])

    # Create a CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Event Name', 'Main Event', 'Country', 'Start Date', 'End Date', 'Total Deaths', 'Total Damage'])  # CSV header

    for result in results:
        writer.writerow([
            result['Event_Name'],
            result['Main_Event'],
            result['Country'],
            result['Start_Date'],
            result['End_Date'],
            result['Total_Deaths'],
            result['Total_Damage']
        ])

    # Reset the pointer to the beginning of the stream
    output.seek(0)

    # Create a response object and set headers for download
    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=search_results.csv'
    response.headers['Content-type'] = 'text/csv'

    return response


if __name__ == '__main__':
    app.run(debug=True)
