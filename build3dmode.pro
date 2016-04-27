
TEMPLATE = app
TARGET = build3dmode
INCLUDEPATH += .

QMAKE_CXXFLAGS = -Wall -g
INCLUDES += `pkg-config --cflags opencv`
LIBS += `pkg-config --libs opencv`

SOURCES += build3dmodel.cpp

