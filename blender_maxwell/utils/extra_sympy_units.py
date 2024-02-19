import sympy as sp
import sympy.physics.units as spu

####################
# - Force
####################
# Newton
nanonewton = nN = spu.Quantity("nanonewton", abbrev="nN")
nanonewton.set_global_relative_scale_factor(spu.nano, spu.newton)

micronewton = uN = spu.Quantity("micronewton", abbrev="μN")
micronewton.set_global_relative_scale_factor(spu.micro, spu.newton)

millinewton = mN = spu.Quantity("micronewton", abbrev="mN")
micronewton.set_global_relative_scale_factor(spu.milli, spu.newton)

####################
# - Frequency
####################
# Hertz
kilohertz = kHz = spu.Quantity("kilohertz", abbrev="kHz")
kilohertz.set_global_relative_scale_factor(spu.kilo, spu.hertz)

megahertz = MHz = spu.Quantity("megahertz", abbrev="MHz")
kilohertz.set_global_relative_scale_factor(spu.kilo, spu.hertz)

gigahertz = GHz = spu.Quantity("gigahertz", abbrev="GHz")
gigahertz.set_global_relative_scale_factor(spu.giga, spu.hertz)

terahertz = THz = spu.Quantity("terahertz", abbrev="THz")
terahertz.set_global_relative_scale_factor(spu.tera, spu.hertz)

petahertz = PHz = spu.Quantity("petahertz", abbrev="PHz")
petahertz.set_global_relative_scale_factor(spu.peta, spu.hertz)

exahertz = EHz = spu.Quantity("exahertz", abbrev="EHz")
exahertz.set_global_relative_scale_factor(spu.exa, spu.hertz)
