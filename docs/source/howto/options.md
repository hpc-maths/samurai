# How-to: set options in samurai

In this how-to guide, we will show you how to set various options in samurai. Options allow you to customize the behavior of samurai components. We use CLI11 library to handle options. Several options are already defined in samurai, but you can also define your own options.

## Setting predefined options

samurai provides several predefined options. You have to parse them to use them. Here is an example of how to use predefined options:

```{literalinclude} snippet/options/predefined_options.cpp
  :language: c++
```

To see the available predefined options, you can run the compiled program with the `--help` flag:

```bash
./your_program --help
```

or

```bash
./your_program -h
```

## Defining custom options

You can also define your own options using the CLI11 library. Here is an example of how to define and use custom options in samurai:

```{literalinclude} snippet/options/custom_options.cpp
  :language: c++
```

For more information on how to use CLI11 to define options, please refer to the [CLI11 documentation](https://cliutils.github.io/CLI11/book/).

## Use a TOML configuration file

With CLI11, you can also use a configuration file to set options. samurai supports TOML configuration files. Here is an example of how to use a TOML configuration file to set options:

```toml
my-option = 2
my-flag = true
```

In this example, we define two options: `my-option` and `my-flag`. The TOML file should be passed to the program using the `--config` flag:

```bash
./your_program --config config.toml
```
