import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:google_sign_in/google_sign_in.dart';

class AccountPage extends StatefulWidget {
  final User user;

  AccountPage(this.user);

  @override
  _AccountPageState createState() => _AccountPageState();
}

class _AccountPageState extends State<AccountPage> {
  final GoogleSignIn _googleSignIn = GoogleSignIn();

  int post = 0;

  @override
  void initState() {
    super.initState();
    FirebaseFirestore.instance.collection('post').where('email', isEqualTo: widget.user.email)
        .get()
        .then((querySnapshot) {
      setState(() {
        post = querySnapshot.docs.length;
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(appBar: _buildAppBar(), body: _buildBody());
  }

  Widget _buildBody() {
    return Container(
      padding: EdgeInsets.all(16.0),
      child: Column(
        children: <Widget>[
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            mainAxisSize: MainAxisSize.max,
            children: <Widget>[
              Column(
                children: <Widget>[
                  Stack(
                    children: <Widget>[
                      SizedBox(
                        width: 80.0,
                        height: 80.0,
                        child: GestureDetector(
                          onTap: () => print('이미지 클릭'),
                          child: CircleAvatar(
                            backgroundImage: NetworkImage(widget.user.photoURL),
                          ),
                        ),
                      ),
                      Container(
                        width: 80.0,
                        height: 80.0,
                        alignment: Alignment.bottomRight,
                        child: Stack(
                          alignment: Alignment.center,
                          children: <Widget>[
                            SizedBox(
                              width: 28.0,
                              height: 28.0,
                              child: FloatingActionButton(
                                onPressed: null,
                                backgroundColor: Colors.white,
                              ),
                            ),
                            SizedBox(
                              width: 25.0,
                              height: 25.0,
                              child: FloatingActionButton(
                                backgroundColor: Colors.blue,
                                onPressed: null,
                                child: Icon(Icons.add),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  Padding(
                    padding: EdgeInsets.all(8.0),
                  ),
                  Text(
                    widget.user.displayName,
                    textAlign: TextAlign.center,
                    style:
                        TextStyle(fontSize: 18.0, fontWeight: FontWeight.bold),
                  )
                ],
              ),
              Padding(
                padding: const EdgeInsets.only(top: 8.0),
                child: Text('$post\n총 라벨수',
                    textAlign: TextAlign.center,
                    style: TextStyle(fontSize: 18.0)),
              ),

            ],
          ),
        ],
      ),
    );
  }

  Widget _buildAppBar() {
    return AppBar(
      actions: <Widget>[
        IconButton(
          icon: Icon(Icons.exit_to_app),
          color: Colors.black,
          onPressed: () {
            FirebaseAuth.instance.signOut();
            _googleSignIn.signOut();
          },
        )
      ],
      backgroundColor: Colors.white,
      title: Text(
        'KM WIFI                                     log out',
        style: TextStyle(color: Colors.black, fontWeight: FontWeight.bold),
      ),
    );
  }
}
