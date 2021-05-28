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
    _logger = logging.getLogger('Server')
    #logger = logging.getLogger("asyncua.address_space")
    #logger = logging.getLogger("asyncua.internal_server")
    #logger = logging.getLogger("asyncua.binary_server_asyncio")
    #logger = logging.getLogger("asyncua.uaprocessor")

    server = Server()
    await server.init()
    server.set_endpoint("opc.tcp://0.0.0.0:16703/")
    server.set_server_name("Servidor OPC eMPC MA")
    await server.set_application_uri(uri="http://servidor-eMPC-MA.com/test/")
    server.set_security_policy([ua.SecurityPolicyType.NoSecurity])
    server._permission_ruleset = None
    server._policyIDs = ["Anonymous"]
    server.certificate = None

    uri = "Servidor OPC eMPC MA"
    idx = await server.register_namespace(uri)

    await server.import_xml("deck_opcua.xml")
    _logger.info("Iniciando servidor OPC-UA...")
    _logger.info("Escuchando en: opc.tcp://localhost:16703/")
    # Crear instancia del controlador
    controlador = clase_MPC.Controlador()
    # starting!
    async with server:
        while True:
            await asyncio.sleep(0.01)
            command_run = await server.get_node("ns=6;s=command_run").get_value()
            if command_run == 1:
                _logger.info(f' [{datetime.now().strftime("%H:%M:%S.%f")[:-3]}]\t Node read: command_run = {command_run}')
                _logger.info(f' [{datetime.now().strftime("%H:%M:%S.%f")[:-3]}]\t Executing...')
                await controlador.recibir_variables(server)
                controlador.actualizar_arrays()
                ControlFlag = await server.get_node("ns=4;s=ControlFlag").read_value()
                _logger.info(f' [{datetime.now().strftime("%H:%M:%S.%f")[:-3]}]\t Node read: Control_Flag = {ControlFlag}')
                
                if ControlFlag:
                    controlador.ejecutar()
                    await server.write_attribute_value(server.get_node("ns=4;s=uq[1]").nodeid,
                                                        ua.DataValue(controlador.uq1))
                    _logger.info(f' [{datetime.now().strftime("%H:%M:%S.%f")[:-3]}]\t \
                                    Node written: uq1 = {controlador.uq1:.3f}')
                    await server.write_attribute_value(server.get_node("ns=4;s=uFr[1]").nodeid,
                                                        ua.DataValue(controlador.uFr1))
                    _logger.info(f' [{datetime.now().strftime("%H:%M:%S.%f")[:-3]}]\t \
                                    Node written: uFr1 = {controlador.uFr1:.3f}')
                    # Falta escribir todas las variables del controlador al servidor
                    await controlador.escribir_variables(server)
                await server.write_attribute_value(server.get_node("ns=6;s=command_run").nodeid, ua.DataValue(0))
                _logger.info(f' [{datetime.now().strftime("%H:%M:%S.%f")[:-3]}]\t Node written: command_run = 0')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
