"""Contains miscellaneous files. console_commands provides basic linux commands (cd, cp, mv, rm).
Currently, these are implemented using os.system: Later this will be changed.
random_number_gen provides seeded generation of integers, uniform distributed floats and normal distributed floats.
mock_data_gen creates peak spectra that can be used for unit tests.
This will probably extended to also test get_power_dependent_data().
misc contains a function that calculates the student-t uncertainty of a mean.
Moreover, functions that handle user input (ints, floats, diameter) are included.
"""