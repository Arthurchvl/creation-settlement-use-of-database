--
-- PostgreSQL database dump
--

-- Dumped from database version 15.13 (Debian 15.13-0+deb12u1)
-- Dumped by pg_dump version 17.0

-- Started on 2025-05-26 11:43:46 CEST

DROP SCHEMA IF EXISTS partie2 CASCADE;
CREATE SCHEMA partie2;
SET SCHEMA 'partie2';

--
-- TOC entry 576 (class 1259 OID 9851484)
-- Name: _individu; Type: TABLE; Schema: partie2; Owner: -
--

CREATE TABLE partie2._individu (
    ine character(11) NOT NULL,
    nom character varying(50) NOT NULL,
    prenom character varying(50) NOT NULL,
    date_naissance date NOT NULL,
    code_postal character varying(10),
    ville character varying(50),
    sexe character varying(50) NOT NULL,
    nationalite character varying(50) NOT NULL
);

--
-- TOC entry 5149 (class 2606 OID 9851488)
-- Name: _individu pk_individu; Type: CONSTRAINT; Schema: partie2; Owner: -
--

ALTER TABLE ONLY partie2._individu
    ADD CONSTRAINT pk_individu PRIMARY KEY (ine);

--
-- TOC entry 578 (class 1259 OID 9851490)
-- Name: _candidat; Type: TABLE; Schema: partie2; Owner: -
--

CREATE TABLE partie2._candidat (
    no_candidat SERIAL NOT NULL,
    classement character varying(100),
    boursier_lycee character varying(100) NOT NULL,
    profil_candidat character varying(100) NOT NULL,
    etablissement character varying(100),
    dept_etablissement character varying(100),
    ville_etablissement character varying(100) NOT NULL,
    niveau_etude character varying(100),
    type_formation_prec character varying(150),
    serie_prec character varying(150),
    dominante_prec character varying(150),
    specialite_prec character varying(150),
    lv1 character varying(100),
    lv2 character varying(100),
    ine character(11) NOT NULL
);

--
-- TOC entry 5151 (class 2606 OID 9851497)
-- Name: _candidat pk_candidat; Type: CONSTRAINT; Schema: partie2; Owner: -
--

ALTER TABLE ONLY partie2._candidat
    ADD CONSTRAINT pk_candidat PRIMARY KEY (no_candidat);

--
-- TOC entry 5164 (class 2606 OID 9851498)
-- Name: _candidat fk_candidat_individu; Type: FK CONSTRAINT; Schema: partie2; Owner: -
--

ALTER TABLE ONLY partie2._candidat
    ADD CONSTRAINT fk_candidat_individu FOREIGN KEY (ine) REFERENCES partie2._individu(ine);


--
-- TOC entry 579 (class 1259 OID 9851503)
-- Name: _etudiant; Type: TABLE; Schema: partie2; Owner: -
--

CREATE TABLE partie2._etudiant (
    code_nip character(11) NOT NULL,
    cat_socio_etud character varying(100) NOT NULL,
    cat_socio_parent character varying(100) NOT NULL,
    bourse_superieur character varying(100),
    mention_bac character varying(100),
    serie_bac character varying(100) NOT NULL,
    dominante_bac character varying(100),
    specialite_bac character varying(100),
    mois_annee_obtention_bac character(40),
    ine character(11) NOT NULL
);

--
-- TOC entry 5153 (class 2606 OID 9851509)
-- Name: _etudiant pk_etudiant; Type: CONSTRAINT; Schema: partie2; Owner: -
--

ALTER TABLE ONLY partie2._etudiant
    ADD CONSTRAINT pk_etudiant PRIMARY KEY (code_nip);

--
-- TOC entry 5165 (class 2606 OID 9851510)
-- Name: _etudiant fk_etudiant_individu; Type: FK CONSTRAINT; Schema: partie2; Owner: -
--

ALTER TABLE ONLY partie2._etudiant
    ADD CONSTRAINT fk_etudiant_individu FOREIGN KEY (ine) REFERENCES partie2._individu(ine);


--
-- TOC entry 580 (class 1259 OID 9851515)
-- Name: _semestre; Type: TABLE; Schema: partie2; Owner: -
--

CREATE TABLE partie2._semestre (
    num_semestre character varying(5) NOT NULL,
    annee_univ character(9) NOT NULL
);

--
-- TOC entry 5155 (class 2606 OID 9851519)
-- Name: _semestre pk_semestre; Type: CONSTRAINT; Schema: partie2; Owner: -
--

ALTER TABLE ONLY partie2._semestre
    ADD CONSTRAINT pk_semestre PRIMARY KEY (num_semestre, annee_univ);



--
-- TOC entry 582 (class 1259 OID 9851525)
-- Name: _inscription; Type: TABLE; Schema: partie2; Owner: -
--

CREATE TABLE partie2._inscription (
    groupe_tp character(2),
    amenagement_evaluation character varying(50),
    code_nip character(11) NOT NULL,
    num_semestre character varying(5) NOT NULL,
    annee_univ character(9) NOT NULL
);

--
-- TOC entry 5159 (class 2606 OID 9851529)
-- Name: _inscription pk_inscription; Type: CONSTRAINT; Schema: partie2; Owner: -
--

ALTER TABLE ONLY partie2._inscription
    ADD CONSTRAINT pk_inscription PRIMARY KEY (code_nip, num_semestre, annee_univ);

--
-- TOC entry 5166 (class 2606 OID 9851530)
-- Name: _inscription fk_inscription_etudiant; Type: FK CONSTRAINT; Schema: partie2; Owner: -
--

ALTER TABLE ONLY partie2._inscription
    ADD CONSTRAINT fk_inscription_etudiant FOREIGN KEY (code_nip) REFERENCES partie2._etudiant(code_nip);


--
-- TOC entry 5167 (class 2606 OID 9851535)
-- Name: _inscription fk_inscription_semestre; Type: FK CONSTRAINT; Schema: partie2; Owner: -
--

ALTER TABLE ONLY partie2._inscription
    ADD CONSTRAINT fk_inscription_semestre FOREIGN KEY (num_semestre, annee_univ) 
            REFERENCES partie2._semestre(num_semestre, annee_univ);

--
-- TOC entry 581 (class 1259 OID 9851520)
-- Name: _module; Type: TABLE; Schema: partie2; Owner: -
--

CREATE TABLE partie2._module (
    id_module character varying(6) NOT NULL,
    libelle_module character varying(150) NOT NULL,
    ue character(4) NOT NULL
);


--
-- TOC entry 5157 (class 2606 OID 9851524)
-- Name: _module pk_module; Type: CONSTRAINT; Schema: partie2; Owner: -
--

ALTER TABLE ONLY partie2._module
    ADD CONSTRAINT pk_module PRIMARY KEY (id_module);

--
-- TOC entry 583 (class 1259 OID 9851540)
-- Name: _programme; Type: TABLE; Schema: partie2; Owner: -
--

CREATE TABLE partie2._programme (
    coefficient numeric(3,1) NOT NULL,
    num_semestre character varying(5) NOT NULL,
    annee_univ character(9) NOT NULL,
    id_module character varying(6) NOT NULL
);

--
-- TOC entry 5161 (class 2606 OID 9851544)
-- Name: _programme pk_programme; Type: CONSTRAINT; Schema: partie2; Owner: -
--

ALTER TABLE ONLY partie2._programme
    ADD CONSTRAINT pk_programme PRIMARY KEY (num_semestre, annee_univ, id_module);

--
-- TOC entry 5168 (class 2606 OID 9851550)
-- Name: _programme fk_programme_module; Type: FK CONSTRAINT; Schema: partie2; Owner: -
--

ALTER TABLE ONLY partie2._programme
    ADD CONSTRAINT fk_programme_module FOREIGN KEY (id_module) 
            REFERENCES partie2._module(id_module);


--
-- TOC entry 5169 (class 2606 OID 9851545)
-- Name: _programme fk_programme_semestre; Type: FK CONSTRAINT; Schema: partie2; Owner: -
--

ALTER TABLE ONLY partie2._programme
    ADD CONSTRAINT fk_programme_semestre FOREIGN KEY (num_semestre, annee_univ) 
            REFERENCES partie2._semestre(num_semestre, annee_univ);

--
-- TOC entry 584 (class 1259 OID 9851555)
-- Name: _resultat; Type: TABLE; Schema: partie2; Owner: -
--

CREATE TABLE partie2._resultat (
    moyenne numeric(4,2) NOT NULL,
    code_nip character(11) NOT NULL,
    num_semestre character varying(5) NOT NULL,
    annee_univ character(9) NOT NULL,
    id_module character varying(6) NOT NULL
);

--
-- TOC entry 5163 (class 2606 OID 9851559)
-- Name: _resultat pk_resultat; Type: CONSTRAINT; Schema: partie2; Owner: -
--

ALTER TABLE ONLY partie2._resultat
    ADD CONSTRAINT pk_resultat PRIMARY KEY (code_nip, num_semestre, annee_univ, id_module);

--
-- TOC entry 5170 (class 2606 OID 9851560)
-- Name: _resultat fk_resultat_etudiant; Type: FK CONSTRAINT; Schema: partie2; Owner: -
--

-- ALTER TABLE ONLY partie2._resultat
--    ADD CONSTRAINT fk_resultat_etudiant FOREIGN KEY (code_nip) 
--            REFERENCES partie2._etudiant(code_nip);

--
-- TOC entry 5171 (class 2606 OID 9851570)
-- Name: _resultat fk_resultat_module; Type: FK CONSTRAINT; Schema: partie2; Owner: -
--

-- ALTER TABLE ONLY partie2._resultat
--    ADD CONSTRAINT fk_resultat_module FOREIGN KEY (id_module) 
--            REFERENCES partie2._module(id_module);


--
-- TOC entry 5172 (class 2606 OID 9851565)
-- Name: _resultat fk_resultat_semestre; Type: FK CONSTRAINT; Schema: partie2; Owner: -
--

-- ALTER TABLE ONLY partie2._resultat
--    ADD CONSTRAINT fk_resultat_semestre FOREIGN KEY (num_semestre, annee_univ) 
--            REFERENCES partie2._semestre(num_semestre, annee_univ);

-- REMPLACE LES DEUX FK vers _etudiant et vers _semestre : pour avoir un résultat, il faut être inscrit.

ALTER TABLE ONLY partie2._resultat 
    ADD CONSTRAINT _resultat_fk_inscription
      FOREIGN KEY (code_nip, annee_univ, num_semestre)
      REFERENCES _inscription(code_nip, annee_univ, num_semestre);

-- En plus, pour avoir un résultat dans un module dans une année, il faut que ce modile soit 
-- au programme cette année-là
-- Pas ebsoin 

ALTER TABLE ONLY partie2._resultat 
    ADD CONSTRAINT _resultat_fk_programme
      FOREIGN KEY (id_module, annee_univ, num_semestre)
      REFERENCES _programme(id_module, annee_univ, num_semestre);

-- Completed on 2025-05-26 11:43:46 CEST

--
-- PostgreSQL database dump complete
--

         
-- import table individu 
WbImport -file=data/v_inscriptions.csv
         -type=text
         -mode=insertIgnore
         -continueOnError=true
         -emptyStringIsNull=true
         -table=partie2._individu
         -fileColumns=$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,ine,nom,prenom,date_naissance,sexe,nationalite,code_postal,ville
         -delimiter=';'
         -header=true
         -keyColumns=ine;
         
WbImport -file=data/v_candidatures.csv
         -type=text
         -mode=insertIgnore
         -continueOnError=true
         -emptyStringIsNull=true
         -table=partie2._individu
         -fileColumns=$wb_skip$,$wb_skip$,$wb_skip$,nom,prenom,sexe,date_naissance,nationalite,code_postal,ville,$wb_skip$,$wb_skip$,$wb_skip$,ine
         -delimiter=';'
         -header=true
         -keyColumns=ine;

-- import table candidat 

-- drop le NOT NULL de la colonne ville_etablissement permet d'enlever des erreurs qui se trouvent dans les fichiers CSV que l'ont ne peut pas résoudre avec la commande        
ALTER TABLE _candidat
    ALTER COLUMN ville_etablissement DROP NOT NULL;
    
WbImport -file=data/v_candidatures.csv
         -type=text
         -mode=insertIgnore
         -continueOnError=true
         -table=partie2._candidat
         -fileColumns=$wb_skip$,no_candidat,classement,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,boursier_lycee,profil_candidat,ine,$wb_skip$,etablissement,ville_etablissement,dept_etablissement,$wb_skip$,niveau_etude,type_formation_prec,serie_prec,dominante_prec,specialite_prec,lv1,lv2
         -delimiter=';'
         -header=true
         -keyColumns=no_candidat;

-- import table etudiant

-- les données à importer se trouvent dans de fichiers CSV différents, il faut donc créer des tables temporaires afin de selectionner les données et de les joindre dans la table finale.
DROP TABLE partie2._etudiant2 CASCADE;
DROP TABLE partie2._etudiant3 CASCADE;
-- drop le NOT NULL de la colonne INE permet d'enlever des erreurs qui se trouvent dans les fichiers CSV que l'ont ne peut pas résoudre avec la commande
ALTER TABLE partie2._etudiant
  ALTER COLUMN ine DROP NOT NULL;
ALTER TABLE partie2._etudiant DROP CONSTRAINT pk_etudiant CASCADE;

CREATE TABLE partie2._etudiant2 AS SELECT * FROM _etudiant;
CREATE TABLE partie2._etudiant3 AS SELECT * FROM _etudiant;

WbImport -file=data/v_inscriptions.csv
         -type=text
         -mode=update,insert
         -continueOnError=true
         -table=partie2._etudiant2
         -fileColumns=$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,code_nip,ine,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,cat_socio_etud,cat_socio_parent,serie_bac,mention_bac,mois_annee_obtention_bac
         -delimiter=';'
         -header=true
         -keyColumns=ine;

WbImport -file=data/v_candidatures.csv
         -type=text
         -mode=update,insert
         -continueOnError=true
         -table=partie2._etudiant3
         -fileColumns=$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,bourse_superieur,$wb_skip$,ine,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,dominante_bac,specialite_bac
         -delimiter=';'
         -header=true
         -keyColumns=ine;
         
INSERT INTO partie2._etudiant
  SELECT e2.code_nip,e2.cat_socio_etud,e2.cat_socio_parent,e3.bourse_superieur,e2.mention_bac,e2.serie_bac,e3.dominante_bac,e3.specialite_bac,e2.mois_annee_obtention_bac,e2.ine
  FROM _etudiant2 e2
  JOIN _etudiant3 e3 ON e2.ine = e3.ine
  WHERE e2.ine = e3.ine
  GROUP BY e2.code_nip,e2.cat_socio_etud,e2.cat_socio_parent,e3.bourse_superieur,e2.mention_bac,e2.serie_bac,e3.dominante_bac,e3.specialite_bac,e2.mois_annee_obtention_bac,e2.ine
  HAVING COUNT(e2.ine) = 1;
  
-- import table semestre
WbImport -file=data/v_resu_s1.csv
         -type=text
         -mode=insertIgnore
         -continueOnError=true
         -table=partie2._semestre
         -fileColumns=annee_univ,num_semestre
         -delimiter=';'
         -header=true;
WbImport -file=data/v_resu_s2.csv
         -type=text
         -mode=insertIgnore
         -continueOnError=true
         -table=partie2._semestre
         -fileColumns=annee_univ,num_semestre
         -delimiter=';'
         -header=true;
WbImport -file=data/v_resu_s3.csv
         -type=text
         -mode=insertIgnore
         -continueOnError=true
         -table=partie2._semestre
         -fileColumns=annee_univ,num_semestre
         -delimiter=';'
         -header=true;
WbImport -file=data/v_resu_s4.csv
         -type=text
         -mode=insertIgnore
         -continueOnError=true
         -table=partie2._semestre
         -fileColumns=annee_univ,num_semestre
         -delimiter=';'
         -header=true;

-- import table inscription
WbImport -file=data/v_resu_s1.csv
         -type=text
         -mode=insertIgnore
         -continueOnError=true
         -table=partie2._inscription
         -fileColumns=annee_univ,num_semestre,code_nip,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,groupe_tp
         -delimiter=';'
         -header=true;
WbImport -file=data/v_resu_s2.csv
         -type=text
         -mode=insertIgnore
         -continueOnError=true
         -table=partie2._inscription
         -fileColumns=annee_univ,num_semestre,code_nip,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,groupe_tp
         -delimiter=';'
         -header=true;
WbImport -file=data/v_resu_s3.csv
         -type=text
         -mode=insertIgnore
         -continueOnError=true
         -table=partie2._inscription
         -fileColumns=annee_univ,num_semestre,code_nip,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,groupe_tp
         -delimiter=';'
         -header=true;
WbImport -file=data/v_resu_s4.csv
         -type=text
         -mode=insertIgnore
         -continueOnError=true
         -table=partie2._inscription
         -fileColumns=annee_univ,num_semestre,code_nip,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,groupe_tp
         -delimiter=';'
         -header=true;
         
-- import de la table module
WbImport -file=data/ppn.csv
         -type=text
         -mode=insertIgnore
         -continueOnError=true
         -table=partie2._module
         -fileColumns=id_module,ue,libelle_module
         -delimiter=';'
         -header=true
         -keycolumns=id_module;

--import table programme
WbImport -file=data/v_programme.csv
         -type=text
         -mode=insertIgnore
         -continueOnError=true
         -table=partie2._programme
         -fileColumns=annee_univ,num_semestre,id_module,coefficient
         -delimiter=';'
         -header=true;
         
--import table resultat

-- les données à importer se trouvent dans de fichiers CSV différents, il faut donc créer des tables temporaires afin de selectionner les données et de les joindre dans la table finale.
DROP TABLE partie2._resu2 CASCADE; 
DROP TABLE partie2._resu3 CASCADE;

CREATE TABLE partie2._resu2 (
    moyenne numeric(4,2) NOT NULL,
    code_nip character(11) NOT NULL,
    num_semestre character varying(5) NOT NULL,
    annee_univ character(9) NOT NULL
);

CREATE TABLE partie2._resu3 (
    num_semestre character varying(5) NOT NULL,
    id_module character varying(6) NOT NULL
);

-- drop le NOT NULL de la colonne moyenne permet d'enlever des erreurs qui se trouvent dans les fichiers CSV que l'ont ne peut pas résoudre avec la commande
ALTER TABLE partie2._resultat
  ALTER COLUMN moyenne DROP NOT NULL;

WbImport -file=data/v_resu_s1.csv
         -type=text
         -mode=insertIgnore
         -continueOnError=true
         -table=partie2._resu2
         -decimal=','
         -fileColumns=annee_univ,num_semestre,code_nip,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,moyenne
         -delimiter=';'
         -header=true
         -keycolumns=code_nip;
WbImport -file=data/v_resu_s2.csv
         -type=text
         -mode=insertIgnore
         -continueOnError=true
         -table=partie2._resu2
         -decimal=','
         -fileColumns=annee_univ,num_semestre,code_nip,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,moyenne
         -delimiter=';'
         -header=true
         -keycolumns=code_nip;
WbImport -file=data/v_resu_s3.csv
         -type=text
         -mode=insertIgnore
         -continueOnError=true
         -table=partie2._resu2
         -decimal=','
         -fileColumns=annee_univ,num_semestre,code_nip,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,moyenne
         -delimiter=';'
         -header=true
         -keycolumns=code_nip;
WbImport -file=data/v_resu_s4.csv
         -type=text
         -mode=insertIgnore
         -continueOnError=true
         -table=partie2._resu2
         -decimal=','
         -fileColumns=annee_univ,num_semestre,code_nip,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,$wb_skip$,moyenne
         -delimiter=';'
         -header=true
         -keycolumns=code_nip;
         
WbImport -file=data/v_programme.csv
         -type=text
         -mode=insertIgnore
         -continueOnError=true
         -table=partie2._resu3
         -fileColumns=$wb_wkip$,num_semestre,id_module
         -delimiter=';'
         -header=true
         -keycolumns=id_module;
         
-- a la suite d'une erreur inexpliquable ni par moi ni par Mr Faure, les commandes ci-dessous ne s'exécutent pas
-- il faut séléctioner chaque commande et l'éxecuter avec CRTL + E sinon cela ne fonctionne pas 
-- utiliser le bouton CTRL + ENTREE reéxécute le dernier WbImport
-- merci de votre compréhension

INSERT INTO partie2._resultat
  SELECT DISTINCT r2.moyenne,r2.code_nip,r2.num_semestre,r2.annee_univ,r3.id_module
  FROM _resu2 r2 
  JOIN _resu3 r3 ON r2.num_semestre = r3.num_semestre 
  WHERE r2.num_semestre = r3.num_semestre;

SELECT * FROM _resultat ORDER BY id_module;
