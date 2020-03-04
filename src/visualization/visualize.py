# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd 
from scipy import ndimage

@click.command()
@click.argument('dataset_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(dataset_filepath, output_filepath):
    """ Runs data processing scripts to turn raw csv data into
        cleaned analyzed data ready to be analyzed by a human (saved in ../visualization).
    """
    plt.style.use('seaborn-deep')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 12

    width, height = plt.figaspect(1.68)
    logger = logging.getLogger(__name__)
    logger.info('making visualizations from raw csv data')
    results = pd.read_csv(dataset_filepath)

    # barlett, kinda clean (4-6 anomalies)
    # bohman, blackmanharris, nuttall = same
    # rolling window:
    # [2.s] 120 clean( 1 anomaly)
    # [1.5] 90 semi clean (3 anomalies).
    # [1.s] 60 semi clean ( 4 - 6 death anomalies)
    # [.5s] 30 ugly (10+)
    res2 = results.groupby(['team', "hero"])["health", "armor", "shield", "death"].rolling(120).mean().reset_index()#.unstack(['team', "hero"])

    res2["frame"] = res2["level_2"] // 12

    res2.loc[res2.death > 0, "death"] = 1 
    res2 = res2.drop("level_2", axis=1)
    res3 = pd.melt(res2, ['team', "hero", "frame", "death"])

    #sns.relplot(x="frame", y="value", hue='variable', col="team", kind="line", data=res3, row="hero")


    fig, axes = plt.subplots(6,2)
    i = 0
    for team in res2.team.unique():
        j = 0
        for hero in res2.hero.unique():
            currentData = res2.loc[(res2.team==team) & (res2.hero == hero)]
            frames = currentData.frame
            health = currentData.health
            shield = currentData.shield
            armor = currentData.armor
            axes[j,i].stackplot(frames, 
                            health, 
                            armor,
                            shield,
                            labels=["health", "armor", "shield"],
                            cmap="Dark2")
            
            j+=1
        i+=1

    handles, labels = axes[5, 1].get_legend_handles_labels()
    lgd= fig.legend(handles, labels,loc='center right')
    fig.savefig(output_filepath+'/health_during_the_game.png',bbox_inches='tight', bbox_extra_artists=(lgd,))




    fig, axes = plt.subplots(6,2)
    i = 0
    for team in res2.team.unique():
        j = 0
        for hero in res2.hero.unique():
            current_data = res2.loc[(res2.team==team) & (res2.hero == hero)]
            frames = current_data.frame
            daed_frames = (current_data.health < 25) & (current_data.death == 1)
            axes[j,i].stackplot(frames,daed_frames, cmap="Accent")
            j+=1
        i+=1
    fig.savefig(output_filepath+'/deaths_during_the_game.png', bbox_inches='tight')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
