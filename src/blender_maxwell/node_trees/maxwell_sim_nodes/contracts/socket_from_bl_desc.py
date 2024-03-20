from .socket_types import SocketType as ST

BL_SOCKET_DESCR_ANNOT_STRING = ':: '
BL_SOCKET_DESCR_TYPE_MAP = {
	('Time', 'NodeSocketFloat', 1): ST.PhysicalTime,
	('Angle', 'NodeSocketFloat', 1): ST.PhysicalAngle,
	('SolidAngle', 'NodeSocketFloat', 1): ST.PhysicalSolidAngle,
	('Rotation', 'NodeSocketVector', 2): ST.PhysicalRot2D,
	('Rotation', 'NodeSocketVector', 3): ST.PhysicalRot3D,
	('Freq', 'NodeSocketFloat', 1): ST.PhysicalFreq,
	('AngFreq', 'NodeSocketFloat', 1): ST.PhysicalAngFreq,
	## Cartesian
	('Length', 'NodeSocketFloat', 1): ST.PhysicalLength,
	('Area', 'NodeSocketFloat', 1): ST.PhysicalArea,
	('Volume', 'NodeSocketFloat', 1): ST.PhysicalVolume,
	('Disp', 'NodeSocketVector', 2): ST.PhysicalDisp2D,
	('Disp', 'NodeSocketVector', 3): ST.PhysicalDisp3D,
	('Point', 'NodeSocketFloat', 1): ST.PhysicalPoint1D,
	('Point', 'NodeSocketVector', 2): ST.PhysicalPoint2D,
	('Point', 'NodeSocketVector', 3): ST.PhysicalPoint3D,
	('Size', 'NodeSocketVector', 2): ST.PhysicalSize2D,
	('Size', 'NodeSocketVector', 3): ST.PhysicalSize3D,
	## Mechanical
	('Mass', 'NodeSocketFloat', 1): ST.PhysicalMass,
	('Speed', 'NodeSocketFloat', 1): ST.PhysicalSpeed,
	('Vel', 'NodeSocketVector', 2): ST.PhysicalVel2D,
	('Vel', 'NodeSocketVector', 3): ST.PhysicalVel3D,
	('Accel', 'NodeSocketFloat', 1): ST.PhysicalAccelScalar,
	('Accel', 'NodeSocketVector', 2): ST.PhysicalAccel2D,
	('Accel', 'NodeSocketVector', 3): ST.PhysicalAccel3D,
	('Force', 'NodeSocketFloat', 1): ST.PhysicalForceScalar,
	('Force', 'NodeSocketVector', 2): ST.PhysicalForce2D,
	('Force', 'NodeSocketVector', 3): ST.PhysicalForce3D,
	('Pressure', 'NodeSocketFloat', 1): ST.PhysicalPressure,
	## Energetic
	('Energy', 'NodeSocketFloat', 1): ST.PhysicalEnergy,
	('Power', 'NodeSocketFloat', 1): ST.PhysicalPower,
	('Temp', 'NodeSocketFloat', 1): ST.PhysicalTemp,
	## ELectrodynamical
	('Curr', 'NodeSocketFloat', 1): ST.PhysicalCurr,
	('CurrDens', 'NodeSocketVector', 2): ST.PhysicalCurrDens2D,
	('CurrDens', 'NodeSocketVector', 3): ST.PhysicalCurrDens3D,
	('Charge', 'NodeSocketFloat', 1): ST.PhysicalCharge,
	('Voltage', 'NodeSocketFloat', 1): ST.PhysicalVoltage,
	('Capacitance', 'NodeSocketFloat', 1): ST.PhysicalCapacitance,
	('Resistance', 'NodeSocketFloat', 1): ST.PhysicalResistance,
	('Conductance', 'NodeSocketFloat', 1): ST.PhysicalConductance,
	('MagFlux', 'NodeSocketFloat', 1): ST.PhysicalMagFlux,
	('MagFluxDens', 'NodeSocketFloat', 1): ST.PhysicalMagFluxDens,
	('Inductance', 'NodeSocketFloat', 1): ST.PhysicalInductance,
	('EField', 'NodeSocketFloat', 2): ST.PhysicalEField3D,
	('EField', 'NodeSocketFloat', 3): ST.PhysicalEField2D,
	('HField', 'NodeSocketFloat', 2): ST.PhysicalHField3D,
	('HField', 'NodeSocketFloat', 3): ST.PhysicalHField2D,
	## Luminal
	('LumIntensity', 'NodeSocketFloat', 1): ST.PhysicalLumIntensity,
	('LumFlux', 'NodeSocketFloat', 1): ST.PhysicalLumFlux,
	('Illuminance', 'NodeSocketFloat', 1): ST.PhysicalIlluminance,
	## Optical
	('PolJones', 'NodeSocketFloat', 2): ST.PhysicalPolJones,
	('Pol', 'NodeSocketFloat', 4): ST.PhysicalPol,
}
