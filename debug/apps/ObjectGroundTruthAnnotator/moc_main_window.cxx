/****************************************************************************
** Meta object code from reading C++ file 'main_window.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.7)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../apps/ObjectGroundTruthAnnotator/main_window.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'main_window.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.7. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_MainWindow[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
      20,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      12,   11,   11,   11, 0x0a,
      28,   11,   11,   11, 0x0a,
      41,   11,   11,   11, 0x0a,
      63,   59,   11,   11, 0x0a,
      95,   11,   11,   11, 0x0a,
     104,   11,   11,   11, 0x0a,
     114,   11,   11,   11, 0x0a,
     123,   11,   11,   11, 0x0a,
     133,   11,   11,   11, 0x0a,
     142,   11,   11,   11, 0x0a,
     152,   11,   11,   11, 0x0a,
     162,   11,   11,   11, 0x0a,
     173,   11,   11,   11, 0x0a,
     183,   11,   11,   11, 0x0a,
     194,   11,   11,   11, 0x0a,
     204,   11,   11,   11, 0x0a,
     215,   11,   11,   11, 0x0a,
     222,   11,   11,   11, 0x0a,
     234,  229,   11,   11, 0x0a,
     267,   11,   11,   11, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_MainWindow[] = {
    "MainWindow\0\0lock_with_icp()\0save_model()\0"
    "remove_selected()\0idx\0"
    "model_list_clicked(QModelIndex)\0"
    "x_plus()\0x_minus()\0y_plus()\0y_minus()\0"
    "z_plus()\0z_minus()\0xr_plus()\0xr_minus()\0"
    "yr_plus()\0yr_minus()\0zr_plus()\0"
    "zr_minus()\0next()\0prev()\0flag\0"
    "enablePoseRefinmentButtons(bool)\0"
    "updateSelectedHypothesis()\0"
};

void MainWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        MainWindow *_t = static_cast<MainWindow *>(_o);
        switch (_id) {
        case 0: _t->lock_with_icp(); break;
        case 1: _t->save_model(); break;
        case 2: _t->remove_selected(); break;
        case 3: _t->model_list_clicked((*reinterpret_cast< const QModelIndex(*)>(_a[1]))); break;
        case 4: _t->x_plus(); break;
        case 5: _t->x_minus(); break;
        case 6: _t->y_plus(); break;
        case 7: _t->y_minus(); break;
        case 8: _t->z_plus(); break;
        case 9: _t->z_minus(); break;
        case 10: _t->xr_plus(); break;
        case 11: _t->xr_minus(); break;
        case 12: _t->yr_plus(); break;
        case 13: _t->yr_minus(); break;
        case 14: _t->zr_plus(); break;
        case 15: _t->zr_minus(); break;
        case 16: _t->next(); break;
        case 17: _t->prev(); break;
        case 18: _t->enablePoseRefinmentButtons((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 19: _t->updateSelectedHypothesis(); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData MainWindow::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject MainWindow::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_MainWindow,
      qt_meta_data_MainWindow, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &MainWindow::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *MainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *MainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_MainWindow))
        return static_cast<void*>(const_cast< MainWindow*>(this));
    return QObject::qt_metacast(_clname);
}

int MainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 20)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 20;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
