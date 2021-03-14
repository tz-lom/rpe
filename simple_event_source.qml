import QtQuick 2.5
import QtQuick.Controls 1.4
import QtQuick.Window 2.2
import QtQuick.Layouts 1.3

import Resonance 3.0 // импортируем Резонанс


ApplicationWindow
{
    id: root    
    width: 800
    height: 600
    visible: true
    title: "Simple event source"
    
    Component.onCompleted: {
        ResonanceApp.setServiceName("simpleEventsSource") // Так можно задать имя сервиса во время исполнения
    }

    MessageSender {
        id: clickSender
        enabled: true // важно 
    }  
        
	Button {
		anchors.fill: parent
		text: "Send events"
		onClicked: {
			clickSender.sendMessage("12:14");
		}
	}
        
}
