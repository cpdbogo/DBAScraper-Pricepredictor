"""Handle CLI GUI/ Main menu."""
import time
from simple_term_menu import TerminalMenu
from scraper import DBAScraper
from machine_learning import machineLearning


class mainPage():
    """Handle CLI GUI/ Main menu."""

    def __init__(self):
        """Initialize scraper and machinelearner."""
        self.scraper = DBAScraper()
        self.trainer = machineLearning()

    def scrape(self):
        """Scrape DBA for Bang & Olufsen ads."""
        self.scraper.scrape_dba()

    def train(self):
        """Train the model for X epochs."""
        print("##################################################################")
        print("#### Please specify a max number of epochs to run in training ####")
        print("####  trainer will stop automatically when loss is no longer  ####")
        print("####                     decreasing                           ####")
        print("##################################################################")
        epochs = input("Epochs to run: ")
        self.trainer.train_or_predict(int(epochs))

    def mainMenu(self):
        """Menu for the application."""
        main_menu_title = "DBA Price predictor for Bang & Olufsen items\n             By Casper P\n"
        main_menu_items = ["Scrape DBA", "Train model", "Predict price", "Quit"]
        main_menu_cursor = "           > "
        main_menu_cursor_style = ("fg_red", "bold")
        main_menu_style = ("bg_blue", "fg_yellow")
        main_menu_exit = False

        main_menu = TerminalMenu(menu_entries=main_menu_items,
                                 title=main_menu_title,
                                 menu_cursor=main_menu_cursor,
                                 menu_cursor_style=main_menu_cursor_style,
                                 menu_highlight_style=main_menu_style,
                                 cycle_cursor=True,
                                 clear_screen=True)
        while not main_menu_exit:
            main_sel = main_menu.show()

            if main_sel == 0:
                self.scrape()
                time.sleep(3)
            elif main_sel == 1:
                self.train()
            elif main_sel == 2:
                self.trainer.predict()
                input("Press enter for main menu")
            elif main_sel == 3:
                main_menu_exit = True


main_menu = mainPage()
main_menu.mainMenu()
