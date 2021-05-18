import asyncio
import copy
import logging
from datetime import datetime
import time
from math import sin
import sys
import Clase_Controlador as clase_MPC
sys.path.insert(0, "..")

from asyncua import ua, uamethod, Server

async def main():
    logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger("asyncua.address_space")
    logger = logging.getLogger("asyncua.internal_server")
    logger = logging.getLogger("asyncua.binary_server_asyncio")
    logger = logging.getLogger("asyncua.uaprocessor")

    server = Server()
    await server.init()
    server.set_endpoint("opc.tcp://127.0.0.1:16703/")
    server.set_server_name("Servidor OPC eMPC MA")
    server.set_security_policy([ua.SecurityPolicyType.NoSecurity])

    uri = "Servidor OPC eMPC MA"
    idx = await server.register_namespace(uri)

    await server.import_xml("deck_opcua.xml")
    print("Iniciando servidor OPC-UA...")
    print("Escuchando en: opc.tcp://localhost:16703/")
    # Crear instancia del controlador
    controlador = clase_MPC.Controlador()
    # starting!
    async with server:
        while True:
            await asyncio.sleep(0.01)
            command_run = server.get_node("ns=6;s=command_run")
            if await command_run.get_value() == 1:
                
                await controlador.recibir_variables(server)
                controlador.actualizar_arrays()
                ControlFlag = await server.get_node("ns=4;s=ControlFlag").read_value()
                if ControlFlag:
                    controlador.ejecutar()
                    print("Acciones de control:")
                    print(f"\t q = {controlador.uq1:.3f}")
                    print(f"\t Fr = {controlador.uFr1:.3f}")
                    await server.write_attribute_value(server.get_node("ns=4;s=uq[1]").nodeid,
                                                        ua.DataValue(controlador.uq1))
                    await server.write_attribute_value(server.get_node("ns=4;s=uFr[1]").nodeid,
                                                        ua.DataValue(controlador.uFr1))
                    # Falta escribir todas las variables del controlador al servidor
                    await controlador.escribir_variables(server)
                await server.write_attribute_value(command_run.nodeid, ua.DataValue(0))

if __name__ == "__main__":
    asyncio.run(main())
