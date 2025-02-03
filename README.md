# Wikimpacts Web Application

Wikimpacts Web is a web-based tool developed by the Department of Water and Climate (HYDR) at the Vrije Universiteit Brussel (VUB) to explore and visualize the impacts of climate-related disasters around the world. The project aims to provide accessible, data-driven insights to support decision-making in disaster risk reduction and climate adaptation strategies.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [License](#license)

## Overview

Wikimpacts Web provides a comprehensive platform that visualizes global impacts of climate-related disasters. It enhances awareness, preparedness, and response efforts among policymakers, researchers, and the public.

## Features

- **Search Functionality**: Allows users to search for events by country, impact type, and events since a specified year.
- **Multiple Visualizations**: Provides different types of visualizations to analyze event data, including time series and geographical maps.
- **Download Data**: Option to download search results in CSV format.
- **Interactive Tables**: Sort and filter data dynamically on the results page.
- **Email Contact**: Integrated contact form to get in touch with the team.

## Setup and Installation

### Prerequisites

- Python 3.9 or higher
- Poetry (for dependency management)

### Installation Steps

1. **Clone the Repository**

    ```sh
    git clone https://github.com/VUB-HYDR/WikimpactsWeb.git
    cd WikimpactsWeb/wikimpacts-web
    ```

2. **Set Up the Virtual Environment with Poetry**

    Install Poetry if you haven't already:
    
    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```

    Install dependencies:
    
    ```sh
    poetry install
    ```

3. **Download `impact.db`**

Make sure the impact.db file is in the main directory of the project. If it's not already there, please add it.

If the `impact.db` file is too large to be stored directly in the repository, download it manually from the [releases page](https://github.com/VUB-HYDR/WikimpactsWeb/releases/tag/impact_database).

    ```sh
    wget -O impact.db https://github.com/VUB-HYDR/WikimpactsWeb/releases/download/impact_database/impact.db
    ```

4. **Run the Application**

    Start the application with Poetry:
    
    ```sh
    poetry run python wikimpacts-web/app.py
    ```

    The application should now be running at `http://127.0.0.1:5000`.

## Usage

### Search

- Navigate to the search page to look up events by country, impact type, or events since a specified year.

### Visualizations

- The visualizations page provides multiple types of visualizations for event data. You can select between time series and maps to explore the data.

### Download Data

- Use the "Download CSV" button on the search results page to export the data in CSV format.

## Project structure
Wikimpacts-web/
├── wikimpacts-web/
│   ├── static/               # Directory for static files (generated)
│   ├── templates/            # HTML templates
│   ├── app.py                # Main Flask application
│   ├── impact.db             # SQLite database (not included in repo, download from releases)
│   ├── Visualization.py      # Visualization functions
├── .gitattributes            # Git LFS configuration
├── .gitignore                # Git ignore file
├── poetry.lock               # Poetry lock file
├── pyproject.toml            # Poetry configuration file
└── README.md                 # Project README file
