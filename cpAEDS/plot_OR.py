from utils import read_output, plot_offset_ratio,load_config_yaml,plot_offset_pH,plot_offset_pH_fraction

settings_loaded = load_config_yaml(
                config= '../final_settings.yaml')
fraction_state1,offset = read_output('./results.out')
plot_offset_pH(offset,fraction_state1,settings_loaded)
plot_offset_pH_fraction(offset,fraction_state1,settings_loaded)
