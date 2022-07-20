import os
from cpaeds.utils import load_config_yaml,get_dir_list,check_finished


if __name__ == "__main__":
        settings_loaded = load_config_yaml(
                config= './final_settings.yaml')

        dir_list = get_dir_list()
        status_list = []
        for dir in dir_list:
                if dir == 'results':
                        continue
                else:
                        os.chdir(f"{settings_loaded['system']['aeds_dir']}/{settings_loaded['system']['output_dir_name']}/{dir}")
                        run_finished, NOMD = check_finished(settings_loaded)
                        status_list.append(run_finished)
        print(status_list)        
        if all(status_list) == True:
                print("Runs finished.")
        else:
                res = list(filter(lambda i: not status_list[i], range(len(status_list))))
                print (f"Run str(res) not finished.")