# Blender Maxwell
*A tool for fast, intuitive, and real-time design and visualization of Maxwell PDE Simulations in Blender 3D*.

**WARNING: This is very alpha software. It says "version 0.0.1" for a reason: Expect janky untest installation; expect obvious bugs, lacking documentation, and jankiness; expect breaking changes without warning. Here be dragons, growl growl...**

Blender Maxwell is a tool for creating and analyzing electromagnetic PDE simulations, with a particular focus on nano-/micro-scale phenomena.
In particular, this tool seeks to make working with FDTD ("Finite-Difference Time-Domain") intuitive and fun:

- **Rapid Node-Based Simulation Design and Visualization**: Rapidly create and visualize complex FDTD simulations with a novel visual programming language, implemented within Blender's (<https://www.blender.org/>) Python API.
- **Deep Tidy3D Integration**: Submit, run, monitor, and retrieve FDTD simulation results from the high-performance, state-of-the-art commercial FDTD solver "Tidy3D" (<https://www.flexcompute.com/tidy3d/solver/>), leveraging their Python client library `tidy3d`.

**Development of this tool is made possible with support from the Technical University of Denmark / DTU Electro / Metamaterials Department. They provided the opportunity to develop the tool as part of the author's BSc Eng thesis, and provided funding for a professional Tidy3D license.**

## Key Features
- **Parametric Design**: Leverage Blender's parametric, non-destructive geometry generator: "Geometry Nodes". <https://docs.blender.org/manual/en/latest/modeling/geometry_nodes/introduction.html>).
- **Live Processing and Visualization**: Quickly understand the nature of data flowing through the node graph, be it a simulation input or output, with the help of (soft-)real-time 2D plots and 3D visualizations tailored to each node.
- **Prioritization of UX**: Much work has gone into node graphs and workflows feeling and acting "right". Here, "right" is an ideal of optimizing for clarity, responsiveness, and features, in roughly that order.

## "Free" in License and Spirit
Blender Maxwell is **free (as in freedom) software** (aka. *open source*).
It endeavors to **discourage** lock-in:
- All simulation components and results are easily imported and exported as open formats, `.json`, `.hdf`, `.png` for plots, etc.. *This is especially possible thanks to the fully open-source `tidy3d` client.*
- Simulation `.blend` files can be **freely shared and used** by absolutely anyone (with the sole exception of *running new simulations on the Tidy3D cloud*, which requires a Tidy3D license + API Key). *This is possible because Blender (and its file format `.blend`) is free (as in freedom) software, meaning the design and results of your work will never be locked up*.
- The entire simulation workspace is scriptable. This makes it easy to write custom scripts to fitting experimental material properties from specialized formats, auto-generating simulation node graphs for educational purposes, and more.
- The license (see License section) enables anyone to ex. to make their own nodes, examine the way certain operations are implemented, audit the source code (to comply with ex. institutional security policies), and/or make similar software by copy-pasting any useful parts of this program.

**We hope you'll consider saying hello in an Issue, reporting bugs, and/or contributing any changes you find useful!**



# Simulator Support
Only Tidy3D is supported, via the `tidy3d` client.

`tidy3d` was chosen as the "lingua franca" client for declarative FDTD simulation definition, in large part because it is "just that good".
Other simulators are theoretically possible to integrate; however, this would involve writing a translation layer from the `tidy3d`.

## Why Tidy3D?
FDTD simulation can easily take days, be very complex to design, and fail in intricate ways.

As what might be the fastest FDTD simulator on the market today, Tidy3D is the "secret sauce" of the rapid iteration loops that Blender Maxwell attempts to help realize.
It turns  **days into minutes** when running useful, real-world simulation tasks.
Moreover, Tidy3D provides a very well-made and -documented open-source Python client library, `tidy3d`, which is a massively helpful resource when developing such a tool.

Due to its quality, performance, and lack of "lock-in", I believe Tidy3D is an ideal and future-proof choice of simulator integration for this project.



# License
This software is provided under the terms of the "AGPL software license", as specified in the `LICENSE` file.
If you would like a different license, please contact the author(s).

In general (*not legal advice*), AGPL means you can **use, modify, redistribute, and even sell (copies of)** this project (or simulation node graphs made with it) for any reason, **so long as you also provide these same kinds of freedoms to anyone you share it (copies) with** (including stating any changes you made).

If you wish to contribute, you must also agree to do so under the terms of this license.
For more information on AGPL, see (<https://www.gnu.org/licenses/gpl-faq.html>) or seek legal advice.

## Do's and Don'ts
It can sometimes be a bit confusing to figure out exactly what you're allowed to do with a piece of software.
If in doubt, don't hesitate to ask.

**Do**: Anything you want with exported simulation output, ex. plots, images, arrays/fields (as ex. HDF, OpenVDB, text, etc.), renders, and so on.
To be very clear, and to dispel a common misunderstanding: AGPL doesn't apply to exported results; exported results are NOT subject to AGPL(-compatible) terms.
In general, when you share something you made **using** Blender Maxwell, AGPL DOESN'T apply; if you share something you made **including** part (or all) of Blender Maxwell, AGPL DOES apply.

**Don't**: Share `.blend` files containing a Blender Maxwell node graph **without an AGPL(-compatible) license**.
Since node graphs are *source code* (visual code is also code) for software that *includes part of Blender Maxwell*, sharing a node graph requires *sharing a part of Blender Maxwell*.
Therefore, the AGPL applies.

**Do**: Take a moment to explore the dependencies, including Blender and `tidy3d` of course, but also `numpy`, `sympy`, `xarray`, `networkx`, `pydantic`, `jax` etc.!
They are not only fantastic pieces of engineering, but are themselves licensed under similar "open source" terms that made this tool possible.

**Don't**: Worry about whether your `.blend` file is licensed correctly.
By default, Blender Maxwell should include license information in the node graph (feel free to change the license, just mind that your choice is AGPL-compatible).
When you do share simulation `.blend`s, it's probably a good idea to briefly communicate the terms anyway, just to everyone's on the same page.
